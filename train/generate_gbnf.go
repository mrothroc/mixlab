package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

const (
	gbnfMaxReadyStacks    = 4_096
	gbnfMaxStackSymbols   = 4_096
	gbnfMaxTokenBytes     = 4_096
	gbnfMaxTokenTrieNodes = 2_000_000
)

type gbnfGrammar struct {
	rules     [][][]int
	terminals []gbnfByteMatcher
	root      int
}

type gbnfState struct {
	grammar *gbnfGrammar
	stacks  [][]int
}

func newGBNFState(grammar *gbnfGrammar) (gbnfState, error) {
	if grammar == nil || grammar.root < 0 || grammar.root >= len(grammar.rules) {
		return gbnfState{}, fmt.Errorf("grammar root is invalid")
	}
	stacks, err := grammar.closeStacks([][]int{{grammar.root}})
	if err != nil {
		return gbnfState{}, err
	}
	return gbnfState{grammar: grammar, stacks: stacks}, nil
}

func (s gbnfState) accepting() bool {
	for _, stack := range s.stacks {
		if len(stack) == 0 {
			return true
		}
	}
	return false
}

func (s gbnfState) consume(value byte) (gbnfState, bool, error) {
	next := make([][]int, 0, len(s.stacks))
	for _, stack := range s.stacks {
		if len(stack) == 0 || stack[0] >= 0 {
			continue
		}
		terminal := -stack[0] - 1
		if terminal < 0 || terminal >= len(s.grammar.terminals) {
			return gbnfState{}, false, fmt.Errorf("grammar terminal index %d is invalid", terminal)
		}
		if s.grammar.terminals[terminal].matches(value) {
			next = append(next, append([]int(nil), stack[1:]...))
		}
	}
	if len(next) == 0 {
		return gbnfState{grammar: s.grammar}, false, nil
	}
	closed, err := s.grammar.closeStacks(next)
	if err != nil {
		return gbnfState{}, false, err
	}
	return gbnfState{grammar: s.grammar, stacks: closed}, len(closed) > 0, nil
}

func (s gbnfState) consumeBytes(values []byte) (gbnfState, bool, error) {
	current := s
	for _, value := range values {
		next, viable, err := current.consume(value)
		if err != nil || !viable {
			return next, viable, err
		}
		current = next
	}
	return current, len(current.stacks) > 0, nil
}

func (g *gbnfGrammar) closeStacks(initial [][]int) ([][]int, error) {
	queue := append([][]int(nil), initial...)
	seen := make(map[string]bool, len(initial))
	ready := make([][]int, 0, len(initial))
	for len(queue) > 0 {
		stack := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if len(stack) > gbnfMaxStackSymbols {
			return nil, fmt.Errorf("grammar expansion exceeds stack-symbol limit %d; left-recursive grammars are unsupported", gbnfMaxStackSymbols)
		}
		key := gbnfStackKey(stack)
		if seen[key] {
			continue
		}
		seen[key] = true
		if len(seen) > gbnfMaxReadyStacks {
			return nil, fmt.Errorf("grammar expansion exceeds stack-state limit %d", gbnfMaxReadyStacks)
		}
		if len(stack) == 0 || stack[0] < 0 {
			ready = append(ready, stack)
			continue
		}
		nonterminal := stack[0]
		if nonterminal >= len(g.rules) {
			return nil, fmt.Errorf("grammar rule index %d is invalid", nonterminal)
		}
		rest := stack[1:]
		for _, production := range g.rules[nonterminal] {
			expanded := make([]int, 0, len(production)+len(rest))
			expanded = append(expanded, production...)
			expanded = append(expanded, rest...)
			queue = append(queue, expanded)
		}
	}
	return ready, nil
}

func gbnfStackKey(stack []int) string {
	if len(stack) == 0 {
		return "e"
	}
	var builder strings.Builder
	for _, symbol := range stack {
		builder.WriteString(strconv.Itoa(symbol))
		builder.WriteByte(',')
	}
	return builder.String()
}

type grammarToken struct {
	bytes   []byte
	special bool
}

type grammarTokenTrie struct {
	children map[byte]*grammarTokenTrie
	tokens   []int
}

type gbnfProcessorSpec struct {
	grammar *gbnfGrammar
	tokens  []grammarToken
	trie    *grammarTokenTrie
	eosID   int
}

type gbnfLogitProcessor struct {
	spec       *gbnfProcessorSpec
	state      gbnfState
	started    bool
	terminated bool
}

func buildGBNFLogitProcessorFactory(source, tokenizerPath, configPath, weightsPath string, cfg *ArchConfig, vocab *data.NucleotideVocabulary, eosTokenID int) (LogitProcessorFactory, error) {
	if strings.TrimSpace(source) == "" {
		return nil, fmt.Errorf("GBNF grammar must not be empty")
	}
	if eosTokenID < 0 {
		return nil, fmt.Errorf("GBNF constrained generation requires -eos-token-id")
	}
	grammar, err := parseGBNF(source)
	if err != nil {
		return nil, fmt.Errorf("parse GBNF grammar: %w", err)
	}
	if _, err := newGBNFState(grammar); err != nil {
		return nil, fmt.Errorf("initialize GBNF grammar: %w", err)
	}
	tokens, err := loadGrammarTokens(tokenizerPath, configPath, weightsPath, cfg.VocabSize, vocab, eosTokenID)
	if err != nil {
		return nil, err
	}
	trie := &grammarTokenTrie{children: make(map[byte]*grammarTokenTrie)}
	trieNodes := 1
	for tokenID, token := range tokens {
		if token.special || tokenID == eosTokenID {
			continue
		}
		if len(token.bytes) == 0 {
			return nil, fmt.Errorf("tokenizer token id %d decodes to no bytes", tokenID)
		}
		if len(token.bytes) > gbnfMaxTokenBytes {
			return nil, fmt.Errorf("tokenizer token id %d has %d bytes, limit is %d", tokenID, len(token.bytes), gbnfMaxTokenBytes)
		}
		node := trie
		for _, value := range token.bytes {
			if node.children[value] == nil {
				node.children[value] = &grammarTokenTrie{children: make(map[byte]*grammarTokenTrie)}
				trieNodes++
				if trieNodes > gbnfMaxTokenTrieNodes {
					return nil, fmt.Errorf("tokenizer byte trie exceeds node limit %d", gbnfMaxTokenTrieNodes)
				}
			}
			node = node.children[value]
		}
		node.tokens = append(node.tokens, tokenID)
	}
	spec := &gbnfProcessorSpec{grammar: grammar, tokens: tokens, trie: trie, eosID: eosTokenID}
	return func() LogitProcessor { return &gbnfLogitProcessor{spec: spec} }, nil
}

func (p *gbnfLogitProcessor) Start(prompt []int) error {
	if p == nil || p.spec == nil {
		return fmt.Errorf("GBNF processor is not initialized")
	}
	state, err := newGBNFState(p.spec.grammar)
	if err != nil {
		return err
	}
	p.state = state
	p.started = true
	p.terminated = false
	for position, token := range prompt {
		if err := p.accept(token, true); err != nil {
			return fmt.Errorf("prompt token %d at position %d: %w", token, position, err)
		}
		if p.terminated && position+1 != len(prompt) {
			return fmt.Errorf("prompt has tokens after EOS at position %d", position)
		}
	}
	return nil
}

func (p *gbnfLogitProcessor) Mask(_ int, logits []float32) error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	if p.terminated {
		return fmt.Errorf("grammar is already terminated")
	}
	if len(logits) != len(p.spec.tokens) {
		return fmt.Errorf("logits size=%d, want vocab_size=%d", len(logits), len(p.spec.tokens))
	}
	allowed := make([]bool, len(logits))
	if p.state.accepting() {
		allowed[p.spec.eosID] = true
	}
	if err := p.walkTokenTrie(p.spec.trie, p.state, allowed); err != nil {
		return err
	}
	count := 0
	for token := range logits {
		if !allowed[token] {
			logits[token] = float32(math.Inf(-1))
		} else {
			count++
		}
	}
	if count == 0 {
		return fmt.Errorf("%w in GBNF state", errGrammarNoContinuation)
	}
	return nil
}

func (p *gbnfLogitProcessor) walkTokenTrie(node *grammarTokenTrie, state gbnfState, allowed []bool) error {
	for _, token := range node.tokens {
		allowed[token] = true
	}
	for value, child := range node.children {
		next, viable, err := state.consume(value)
		if err != nil {
			return err
		}
		if viable {
			if err := p.walkTokenTrie(child, next, allowed); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *gbnfLogitProcessor) Accept(token int) error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	return p.accept(token, false)
}

func (p *gbnfLogitProcessor) Finish() error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	if p.terminated || p.state.accepting() {
		return nil
	}
	return fmt.Errorf("GBNF grammar is not accepting")
}

func (p *gbnfLogitProcessor) accept(token int, prompt bool) error {
	if token < 0 || token >= len(p.spec.tokens) {
		return fmt.Errorf("token %d is outside [0,%d)", token, len(p.spec.tokens))
	}
	if token == p.spec.eosID {
		if !p.state.accepting() {
			return fmt.Errorf("EOS token %d is not legal before the grammar accepts", token)
		}
		p.terminated = true
		return nil
	}
	descriptor := p.spec.tokens[token]
	if descriptor.special {
		if prompt {
			return nil
		}
		return fmt.Errorf("special token %d is not legal grammar output", token)
	}
	next, viable, err := p.state.consumeBytes(descriptor.bytes)
	if err != nil {
		return err
	}
	if !viable {
		return fmt.Errorf("token %d does not continue the grammar", token)
	}
	p.state = next
	return nil
}

func loadGrammarTokens(explicit, configPath, weightsPath string, vocabSize int, vocab *data.NucleotideVocabulary, eosTokenID int) ([]grammarToken, error) {
	if vocab != nil {
		tokens := make([]grammarToken, vocab.Size())
		for surface, id := range vocab.Tokens {
			special := strings.HasPrefix(surface, "<") && strings.HasSuffix(surface, ">")
			tokens[id] = grammarToken{bytes: []byte(surface), special: special}
		}
		tokens[eosTokenID].special = true
		return tokens, nil
	}
	path := explicit
	if path == "" {
		if discovered, ok := discoverTokenizerJSON(configPath, weightsPath); ok {
			path = discovered
		}
	}
	if path == "" {
		return nil, fmt.Errorf("GBNF generation requires -tokenizer-path pointing at a ByteLevel BPE tokenizer.json")
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("verify tokenizer path %q: %w", path, err)
	}
	if info.IsDir() {
		path = filepath.Join(path, "tokenizer.json")
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer %q: %w", path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var doc map[string]any
	if err := decoder.Decode(&doc); err != nil {
		return nil, fmt.Errorf("parse tokenizer %q: %w", path, err)
	}
	model, ok := doc["model"].(map[string]any)
	if !ok || !strings.EqualFold(stringValue(model["type"]), "BPE") || findTokenizerNode(doc["pre_tokenizer"], "ByteLevel") == nil {
		return nil, fmt.Errorf("GBNF generation requires a Hugging Face ByteLevel BPE tokenizer")
	}
	surfaces, err := tokenizerVocabularyByID(model, vocabSize)
	if err != nil {
		return nil, fmt.Errorf("tokenizer %q: %w", path, err)
	}
	special := make([]bool, vocabSize)
	addedRaw := make([]bool, vocabSize)
	if entries, ok := doc["added_tokens"].([]any); ok {
		for i, raw := range entries {
			entry, ok := raw.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("tokenizer %q added_tokens[%d] is not an object", path, i)
			}
			id, ok := integerValue(entry["id"])
			if !ok || id < 0 || id >= vocabSize {
				return nil, fmt.Errorf("tokenizer %q added_tokens[%d] has invalid id", path, i)
			}
			if surfaces[id] == "" {
				surfaces[id] = stringValue(entry["content"])
			}
			if value, ok := entry["special"].(bool); ok && value {
				special[id] = true
			} else {
				addedRaw[id] = true
			}
		}
	}
	reverse := byteLevelReverseMap()
	tokens := make([]grammarToken, vocabSize)
	for id, surface := range surfaces {
		if surface == "" {
			return nil, fmt.Errorf("tokenizer %q is missing token id %d", path, id)
		}
		if special[id] {
			tokens[id] = grammarToken{special: true}
			continue
		}
		if addedRaw[id] {
			tokens[id] = grammarToken{bytes: []byte(surface)}
			continue
		}
		decoded := make([]byte, 0, len(surface))
		for _, value := range surface {
			byteValue, ok := reverse[value]
			if !ok {
				return nil, fmt.Errorf("tokenizer %q token id %d contains non-ByteLevel rune %U", path, id, value)
			}
			decoded = append(decoded, byteValue)
		}
		tokens[id] = grammarToken{bytes: decoded}
	}
	tokens[eosTokenID].special = true
	return tokens, nil
}
