package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

const (
	tokenDFAFormat         = "mixlab.token_dfa"
	tokenDFAVersion        = 1
	maxTokenDFAStates      = 100_000
	maxTokenDFATransitions = 1_000_000
	maxTokenDFABytes       = 64 << 20
)

type tokenDFATableJSON struct {
	Format      string              `json:"format"`
	Version     int                 `json:"version"`
	VocabSize   int                 `json:"vocab_size"`
	StartState  string              `json:"start_state"`
	EOSTokenIDs []int               `json:"eos_token_ids"`
	States      []tokenDFAStateJSON `json:"states"`
}

type tokenDFAStateJSON struct {
	Name        string            `json:"name"`
	Accept      bool              `json:"accept,omitempty"`
	Transitions map[string]string `json:"transitions,omitempty"`
}

type tokenDFA struct {
	vocabSize   int
	startState  int
	states      []tokenDFAState
	eosTokenIDs map[int]bool
}

type tokenDFAState struct {
	name        string
	accept      bool
	transitions map[int]int
}

func loadTokenDFA(path string, vocabSize, eosTokenID int) (*tokenDFA, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read grammar table %q: %w", path, err)
	}
	if len(body) > maxTokenDFABytes {
		return nil, fmt.Errorf("grammar table %q has %d bytes, limit is %d", path, len(body), maxTokenDFABytes)
	}
	var table tokenDFATableJSON
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&table); err != nil {
		return nil, fmt.Errorf("parse grammar table %q: %w", path, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return nil, fmt.Errorf("parse grammar table %q: %w", path, err)
	}
	compiled, err := compileTokenDFA(table, vocabSize, eosTokenID)
	if err != nil {
		return nil, fmt.Errorf("grammar table %q: %w", path, err)
	}
	return compiled, nil
}

func compileTokenDFA(table tokenDFATableJSON, vocabSize, eosTokenID int) (*tokenDFA, error) {
	if table.Format != tokenDFAFormat {
		return nil, fmt.Errorf("format=%q, want %q", table.Format, tokenDFAFormat)
	}
	if table.Version != tokenDFAVersion {
		return nil, fmt.Errorf("version=%d is unsupported; this build supports version %d", table.Version, tokenDFAVersion)
	}
	if table.VocabSize != vocabSize {
		return nil, fmt.Errorf("vocab_size=%d does not match model vocab_size=%d", table.VocabSize, vocabSize)
	}
	if len(table.States) == 0 || len(table.States) > maxTokenDFAStates {
		return nil, fmt.Errorf("states count=%d must be in [1,%d]", len(table.States), maxTokenDFAStates)
	}
	stateIndex := make(map[string]int, len(table.States))
	states := make([]tokenDFAState, len(table.States))
	for i, state := range table.States {
		name := strings.TrimSpace(state.Name)
		if name == "" {
			return nil, fmt.Errorf("states[%d].name must not be empty", i)
		}
		if _, exists := stateIndex[name]; exists {
			return nil, fmt.Errorf("duplicate state name %q", name)
		}
		stateIndex[name] = i
		states[i] = tokenDFAState{name: name, accept: state.Accept, transitions: make(map[int]int, len(state.Transitions))}
	}
	startState, ok := stateIndex[table.StartState]
	if !ok {
		return nil, fmt.Errorf("start_state=%q is not declared", table.StartState)
	}
	transitionCount := 0
	acceptCount := 0
	eosTransitionCount := make(map[int]int, len(table.EOSTokenIDs))
	for i, state := range table.States {
		if state.Accept {
			acceptCount++
		}
		for rawToken, nextName := range state.Transitions {
			token, err := strconv.Atoi(rawToken)
			if err != nil || strconv.Itoa(token) != rawToken {
				return nil, fmt.Errorf("state %q transition token %q must be a canonical decimal token id", state.Name, rawToken)
			}
			if token < 0 || token >= vocabSize {
				return nil, fmt.Errorf("state %q transition token %d is outside [0,%d)", state.Name, token, vocabSize)
			}
			next, exists := stateIndex[nextName]
			if !exists {
				return nil, fmt.Errorf("state %q transition token %d targets undeclared state %q", state.Name, token, nextName)
			}
			states[i].transitions[token] = next
			for _, eosToken := range table.EOSTokenIDs {
				if token == eosToken {
					eosTransitionCount[token]++
				}
			}
			transitionCount++
			if transitionCount > maxTokenDFATransitions {
				return nil, fmt.Errorf("transition count exceeds limit %d", maxTokenDFATransitions)
			}
		}
	}
	if acceptCount == 0 {
		return nil, fmt.Errorf("at least one state must set accept=true")
	}
	if len(table.EOSTokenIDs) == 0 {
		return nil, fmt.Errorf("eos_token_ids must contain at least one token id")
	}
	eosIDs := make(map[int]bool, len(table.EOSTokenIDs))
	for _, token := range table.EOSTokenIDs {
		if token < 0 || token >= vocabSize {
			return nil, fmt.Errorf("eos token %d is outside [0,%d)", token, vocabSize)
		}
		if eosIDs[token] {
			return nil, fmt.Errorf("duplicate eos token id %d", token)
		}
		eosIDs[token] = true
	}
	if eosTokenID < 0 {
		return nil, fmt.Errorf("constrained generation requires -eos-token-id to match one of eos_token_ids")
	}
	if !eosIDs[eosTokenID] {
		return nil, fmt.Errorf("-eos-token-id=%d is not listed in eos_token_ids", eosTokenID)
	}
	for _, state := range states {
		for token, next := range state.transitions {
			if eosIDs[token] && !states[next].accept {
				return nil, fmt.Errorf("state %q EOS transition token %d must target an accepting state", state.name, token)
			}
		}
	}
	for token := range eosIDs {
		if eosTransitionCount[token] == 0 {
			return nil, fmt.Errorf("eos token %d has no explicit transition", token)
		}
	}
	return &tokenDFA{vocabSize: vocabSize, startState: startState, states: states, eosTokenIDs: eosIDs}, nil
}

type tokenDFAProcessor struct {
	dfa     *tokenDFA
	state   int
	started bool
}

func (p *tokenDFAProcessor) Start(prompt []int) error {
	if p == nil || p.dfa == nil {
		return fmt.Errorf("token DFA processor is not initialized")
	}
	p.state = p.dfa.startState
	p.started = true
	for pos, token := range prompt {
		if err := p.advance(token); err != nil {
			return fmt.Errorf("prompt token %d at position %d: %w", token, pos, err)
		}
	}
	return nil
}

func (p *tokenDFAProcessor) Mask(_ int, logits []float32) error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	if len(logits) != p.dfa.vocabSize {
		return fmt.Errorf("logits size=%d, want vocab_size=%d", len(logits), p.dfa.vocabSize)
	}
	allowed := p.dfa.states[p.state].transitions
	if len(allowed) == 0 {
		return fmt.Errorf("%w from state %q", errGrammarNoContinuation, p.dfa.states[p.state].name)
	}
	for token := range logits {
		if _, ok := allowed[token]; !ok {
			logits[token] = float32(math.Inf(-1))
		}
	}
	return nil
}

func (p *tokenDFAProcessor) Accept(token int) error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	return p.advance(token)
}

func (p *tokenDFAProcessor) Finish() error {
	if !p.started {
		return fmt.Errorf("processor has not been started")
	}
	if !p.dfa.states[p.state].accept {
		return fmt.Errorf("state %q is not accepting", p.dfa.states[p.state].name)
	}
	return nil
}

func (p *tokenDFAProcessor) advance(token int) error {
	if token < 0 || token >= p.dfa.vocabSize {
		return fmt.Errorf("token %d is outside [0,%d)", token, p.dfa.vocabSize)
	}
	next, ok := p.dfa.states[p.state].transitions[token]
	if !ok {
		return fmt.Errorf("token %d is not legal in state %q", token, p.dfa.states[p.state].name)
	}
	p.state = next
	return nil
}
