package train

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	gbnfMaxSourceBytes = 4 << 20
	gbnfMaxRules       = 10_000
	gbnfMaxProductions = 100_000
	gbnfMaxTerminals   = 1_000_000
	gbnfMaxRepeat      = 1_024
	// gbnfMaxNestingDepth bounds parser recursion over parenthesized groups so
	// an adversarial grammar of deeply nested "(" cannot overflow the goroutine
	// stack (an unrecoverable fatal error) before any ")" is required.
	gbnfMaxNestingDepth = 1_000
)

type gbnfNodeKind uint8

const (
	gbnfNodeEmpty gbnfNodeKind = iota
	gbnfNodeTerminal
	gbnfNodeReference
	gbnfNodeSequence
	gbnfNodeAlternate
	gbnfNodeRepeat
)

type gbnfNode struct {
	kind     gbnfNodeKind
	matcher  gbnfByteMatcher
	name     string
	children []*gbnfNode
	min      int
	max      int // -1 means unbounded.
}

type gbnfByteMatcher [4]uint64

func (m *gbnfByteMatcher) add(value byte) {
	m[value/64] |= uint64(1) << (value % 64)
}

func (m gbnfByteMatcher) matches(value byte) bool {
	return m[value/64]&(uint64(1)<<(value%64)) != 0
}

func gbnfExactByte(value byte) gbnfByteMatcher {
	var matcher gbnfByteMatcher
	matcher.add(value)
	return matcher
}

func gbnfAnyByte() gbnfByteMatcher {
	return gbnfByteMatcher{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
}

type gbnfExpressionParser struct {
	input string
	pos   int
	depth int
}

func parseGBNF(source string) (*gbnfGrammar, error) {
	if len(source) > gbnfMaxSourceBytes {
		return nil, fmt.Errorf("grammar source has %d bytes, limit is %d", len(source), gbnfMaxSourceBytes)
	}
	rules, order, err := splitGBNFRules(source)
	if err != nil {
		return nil, err
	}
	if len(rules) > gbnfMaxRules {
		return nil, fmt.Errorf("grammar has %d rules, limit is %d", len(rules), gbnfMaxRules)
	}
	parsed := make(map[string]*gbnfNode, len(rules))
	for _, name := range order {
		parser := gbnfExpressionParser{input: rules[name]}
		node, err := parser.parseAlternate()
		if err != nil {
			return nil, fmt.Errorf("rule %q: %w", name, err)
		}
		parser.skipSpace()
		if parser.pos != len(parser.input) {
			return nil, fmt.Errorf("rule %q: unexpected %q at byte %d", name, parser.input[parser.pos:], parser.pos)
		}
		parsed[name] = node
	}
	compiler := newGBNFCompiler(order)
	for _, name := range order {
		if err := compiler.compileNamed(name, parsed[name]); err != nil {
			return nil, fmt.Errorf("rule %q: %w", name, err)
		}
	}
	if err := compiler.validateReferences(); err != nil {
		return nil, err
	}
	root, ok := compiler.named["root"]
	if !ok {
		return nil, fmt.Errorf("grammar must define a root rule")
	}
	if compiler.productionCount > gbnfMaxProductions {
		return nil, fmt.Errorf("compiled grammar has %d productions, limit is %d", compiler.productionCount, gbnfMaxProductions)
	}
	if len(compiler.rules) > gbnfMaxRules {
		return nil, fmt.Errorf("compiled grammar has %d rules, limit is %d", len(compiler.rules), gbnfMaxRules)
	}
	if len(compiler.terminals) > gbnfMaxTerminals {
		return nil, fmt.Errorf("compiled grammar has %d terminals, limit is %d", len(compiler.terminals), gbnfMaxTerminals)
	}
	return &gbnfGrammar{rules: compiler.rules, terminals: compiler.terminals, root: root}, nil
}

func splitGBNFRules(source string) (map[string]string, []string, error) {
	rules := make(map[string]string)
	order := make([]string, 0)
	current := ""
	for lineNumber, raw := range strings.Split(source, "\n") {
		line := strings.TrimSpace(stripGBNFComment(raw))
		if line == "" {
			continue
		}
		if index := gbnfDefinitionIndex(line); index >= 0 {
			name := strings.TrimSpace(line[:index])
			if !validGBNFRuleName(name) {
				return nil, nil, fmt.Errorf("line %d has invalid rule name %q", lineNumber+1, name)
			}
			if _, exists := rules[name]; exists {
				return nil, nil, fmt.Errorf("line %d redeclares rule %q", lineNumber+1, name)
			}
			current = name
			rules[name] = strings.TrimSpace(line[index+3:])
			order = append(order, name)
			continue
		}
		if current == "" {
			return nil, nil, fmt.Errorf("line %d is not a rule definition", lineNumber+1)
		}
		rules[current] += " " + line
	}
	if len(rules) == 0 {
		return nil, nil, fmt.Errorf("grammar contains no rules")
	}
	return rules, order, nil
}

func gbnfDefinitionIndex(line string) int {
	inString := false
	inClass := false
	escaped := false
	for index := 0; index < len(line); index++ {
		value := line[index]
		if escaped {
			escaped = false
			continue
		}
		if value == '\\' && (inString || inClass) {
			escaped = true
			continue
		}
		switch value {
		case '"':
			if !inClass {
				inString = !inString
			}
		case '[':
			if !inString {
				inClass = true
			}
		case ']':
			if !inString {
				inClass = false
			}
		case ':':
			if !inString && !inClass && strings.HasPrefix(line[index:], "::=") {
				return index
			}
		}
	}
	return -1
}

func stripGBNFComment(line string) string {
	inString := false
	inClass := false
	escaped := false
	for i, r := range line {
		if escaped {
			escaped = false
			continue
		}
		if r == '\\' && (inString || inClass) {
			escaped = true
			continue
		}
		switch r {
		case '"':
			if !inClass {
				inString = !inString
			}
		case '[':
			if !inString {
				inClass = true
			}
		case ']':
			if !inString {
				inClass = false
			}
		case '#':
			if !inString && !inClass {
				return line[:i]
			}
		}
	}
	return line
}

func validGBNFRuleName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		valid := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || (i > 0 && (r == '-' || (r >= '0' && r <= '9')))
		if !valid {
			return false
		}
	}
	return true
}

func (p *gbnfExpressionParser) parseAlternate() (*gbnfNode, error) {
	p.depth++
	if p.depth > gbnfMaxNestingDepth {
		return nil, fmt.Errorf("grammar nesting exceeds depth limit %d", gbnfMaxNestingDepth)
	}
	defer func() { p.depth-- }()
	parts := make([]*gbnfNode, 0, 2)
	for {
		sequence, err := p.parseSequence()
		if err != nil {
			return nil, err
		}
		parts = append(parts, sequence)
		p.skipSpace()
		if !p.take('|') {
			break
		}
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return &gbnfNode{kind: gbnfNodeAlternate, children: parts}, nil
}

func (p *gbnfExpressionParser) parseSequence() (*gbnfNode, error) {
	parts := make([]*gbnfNode, 0, 4)
	for {
		p.skipSpace()
		if p.pos >= len(p.input) || p.input[p.pos] == '|' || p.input[p.pos] == ')' {
			break
		}
		primary, err := p.parsePrimary()
		if err != nil {
			return nil, err
		}
		primary, err = p.parseQuantifier(primary)
		if err != nil {
			return nil, err
		}
		parts = append(parts, primary)
	}
	switch len(parts) {
	case 0:
		return &gbnfNode{kind: gbnfNodeEmpty}, nil
	case 1:
		return parts[0], nil
	default:
		return &gbnfNode{kind: gbnfNodeSequence, children: parts}, nil
	}
}

func (p *gbnfExpressionParser) parsePrimary() (*gbnfNode, error) {
	p.skipSpace()
	if p.pos >= len(p.input) {
		return nil, fmt.Errorf("expected expression")
	}
	switch p.input[p.pos] {
	case '"':
		return p.parseLiteral()
	case '[':
		return p.parseClass()
	case '.':
		p.pos++
		return &gbnfNode{kind: gbnfNodeTerminal, matcher: gbnfAnyByte()}, nil
	case '(':
		p.pos++
		node, err := p.parseAlternate()
		if err != nil {
			return nil, err
		}
		p.skipSpace()
		if !p.take(')') {
			return nil, fmt.Errorf("unclosed group at byte %d", p.pos)
		}
		return node, nil
	default:
		start := p.pos
		for p.pos < len(p.input) {
			value := p.input[p.pos]
			if (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || value == '_' || (p.pos > start && (value == '-' || (value >= '0' && value <= '9'))) {
				p.pos++
				continue
			}
			break
		}
		if start == p.pos {
			return nil, fmt.Errorf("unexpected character %q at byte %d", p.input[p.pos], p.pos)
		}
		return &gbnfNode{kind: gbnfNodeReference, name: p.input[start:p.pos]}, nil
	}
}

func (p *gbnfExpressionParser) parseLiteral() (*gbnfNode, error) {
	p.pos++
	values := make([]byte, 0)
	for p.pos < len(p.input) && p.input[p.pos] != '"' {
		if p.input[p.pos] == '\\' {
			decoded, err := p.parseEscape(false)
			if err != nil {
				return nil, err
			}
			values = append(values, decoded...)
			continue
		}
		r, size := utf8.DecodeRuneInString(p.input[p.pos:])
		if r == utf8.RuneError && size == 1 {
			return nil, fmt.Errorf("invalid UTF-8 in literal at byte %d", p.pos)
		}
		values = append(values, p.input[p.pos:p.pos+size]...)
		p.pos += size
	}
	if !p.take('"') {
		return nil, fmt.Errorf("unclosed string literal")
	}
	if len(values) == 0 {
		return &gbnfNode{kind: gbnfNodeEmpty}, nil
	}
	children := make([]*gbnfNode, len(values))
	for i, value := range values {
		children[i] = &gbnfNode{kind: gbnfNodeTerminal, matcher: gbnfExactByte(value)}
	}
	if len(children) == 1 {
		return children[0], nil
	}
	return &gbnfNode{kind: gbnfNodeSequence, children: children}, nil
}

func (p *gbnfExpressionParser) parseClass() (*gbnfNode, error) {
	p.pos++
	negated := p.take('^')
	var matcher gbnfByteMatcher
	have := false
	for p.pos < len(p.input) && p.input[p.pos] != ']' {
		start, err := p.parseClassByte()
		if err != nil {
			return nil, err
		}
		end := start
		if p.pos < len(p.input)-1 && p.input[p.pos] == '-' && p.input[p.pos+1] != ']' {
			p.pos++
			end, err = p.parseClassByte()
			if err != nil {
				return nil, err
			}
			if end < start {
				return nil, fmt.Errorf("descending character-class range %d-%d", start, end)
			}
		}
		for value := int(start); value <= int(end); value++ {
			matcher.add(byte(value))
		}
		have = true
	}
	if !p.take(']') {
		return nil, fmt.Errorf("unclosed character class")
	}
	if !have {
		return nil, fmt.Errorf("empty character class")
	}
	if negated {
		for i := range matcher {
			matcher[i] = ^matcher[i]
		}
	}
	return &gbnfNode{kind: gbnfNodeTerminal, matcher: matcher}, nil
}

func (p *gbnfExpressionParser) parseClassByte() (byte, error) {
	if p.pos >= len(p.input) {
		return 0, fmt.Errorf("unclosed character class")
	}
	if p.input[p.pos] == '\\' {
		value, err := p.parseEscape(true)
		if err != nil {
			return 0, err
		}
		if len(value) != 1 {
			return 0, fmt.Errorf("character classes support byte-valued escapes only")
		}
		return value[0], nil
	}
	r, size := utf8.DecodeRuneInString(p.input[p.pos:])
	if r > 0x7f || size != 1 {
		return 0, fmt.Errorf("character classes support ASCII bytes only; use a string literal for %q", r)
	}
	p.pos++
	return byte(r), nil
}

func (p *gbnfExpressionParser) parseEscape(class bool) ([]byte, error) {
	start := p.pos
	p.pos++
	if p.pos >= len(p.input) {
		return nil, fmt.Errorf("trailing escape at byte %d", start)
	}
	value := p.input[p.pos]
	p.pos++
	switch value {
	case 'n':
		return []byte{'\n'}, nil
	case 'r':
		return []byte{'\r'}, nil
	case 't':
		return []byte{'\t'}, nil
	case '"', '\\', '[', ']', '-', '^':
		return []byte{value}, nil
	case 'x':
		return p.parseByteEscape()
	case 'u':
		return p.parseHexEscape(4, class)
	case 'U':
		return p.parseHexEscape(8, class)
	default:
		return nil, fmt.Errorf("unsupported escape \\%c at byte %d", value, start)
	}
}

func (p *gbnfExpressionParser) parseByteEscape() ([]byte, error) {
	if p.pos+2 > len(p.input) {
		return nil, fmt.Errorf("truncated hexadecimal byte escape")
	}
	value, err := strconv.ParseUint(p.input[p.pos:p.pos+2], 16, 8)
	if err != nil {
		return nil, fmt.Errorf("invalid hexadecimal byte escape: %w", err)
	}
	p.pos += 2
	return []byte{byte(value)}, nil
}

func (p *gbnfExpressionParser) parseHexEscape(digits int, class bool) ([]byte, error) {
	if p.pos+digits > len(p.input) {
		return nil, fmt.Errorf("truncated hexadecimal escape")
	}
	value, err := strconv.ParseUint(p.input[p.pos:p.pos+digits], 16, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid hexadecimal escape: %w", err)
	}
	p.pos += digits
	if class {
		if value > 255 {
			return nil, fmt.Errorf("character classes support byte values only, got U+%04X", value)
		}
		return []byte{byte(value)}, nil
	}
	if value > utf8.MaxRune || value >= 0xd800 && value <= 0xdfff {
		return nil, fmt.Errorf("invalid Unicode code point U+%X", value)
	}
	buffer := make([]byte, utf8.RuneLen(rune(value)))
	utf8.EncodeRune(buffer, rune(value))
	return buffer, nil
}

func (p *gbnfExpressionParser) parseQuantifier(node *gbnfNode) (*gbnfNode, error) {
	p.skipSpace()
	if p.pos >= len(p.input) {
		return node, nil
	}
	switch p.input[p.pos] {
	case '?':
		p.pos++
		return &gbnfNode{kind: gbnfNodeRepeat, children: []*gbnfNode{node}, min: 0, max: 1}, nil
	case '*':
		p.pos++
		return &gbnfNode{kind: gbnfNodeRepeat, children: []*gbnfNode{node}, min: 0, max: -1}, nil
	case '+':
		p.pos++
		return &gbnfNode{kind: gbnfNodeRepeat, children: []*gbnfNode{node}, min: 1, max: -1}, nil
	case '{':
		p.pos++
		p.skipSpace()
		minimum, ok := p.parseDecimal()
		if !ok {
			return nil, fmt.Errorf("repeat at byte %d requires a minimum", p.pos)
		}
		p.skipSpace()
		maximum := minimum
		if p.take(',') {
			p.skipSpace()
			if value, present := p.parseDecimal(); present {
				maximum = value
			} else {
				maximum = -1
			}
			p.skipSpace()
		}
		if !p.take('}') {
			return nil, fmt.Errorf("unclosed repeat at byte %d", p.pos)
		}
		if minimum > gbnfMaxRepeat || maximum > gbnfMaxRepeat || maximum >= 0 && maximum < minimum {
			return nil, fmt.Errorf("repeat {%d,%d} is invalid or exceeds limit %d", minimum, maximum, gbnfMaxRepeat)
		}
		return &gbnfNode{kind: gbnfNodeRepeat, children: []*gbnfNode{node}, min: minimum, max: maximum}, nil
	default:
		return node, nil
	}
}

func (p *gbnfExpressionParser) parseDecimal() (int, bool) {
	start := p.pos
	for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
		p.pos++
	}
	if start == p.pos {
		return 0, false
	}
	value, err := strconv.Atoi(p.input[start:p.pos])
	if err != nil {
		// Overflow (or any parse failure): return a value that the caller's
		// gbnfMaxRepeat check rejects rather than silently collapsing to 0.
		return gbnfMaxRepeat + 1, true
	}
	return value, true
}

func (p *gbnfExpressionParser) skipSpace() {
	for p.pos < len(p.input) && (p.input[p.pos] == ' ' || p.input[p.pos] == '\t' || p.input[p.pos] == '\r' || p.input[p.pos] == '\n') {
		p.pos++
	}
}

func (p *gbnfExpressionParser) take(value byte) bool {
	if p.pos < len(p.input) && p.input[p.pos] == value {
		p.pos++
		return true
	}
	return false
}

type gbnfCompiler struct {
	named           map[string]int
	rules           [][][]int
	terminals       []gbnfByteMatcher
	references      map[string]bool
	productionCount int
}

func newGBNFCompiler(names []string) *gbnfCompiler {
	named := make(map[string]int, len(names))
	for i, name := range names {
		named[name] = i
	}
	return &gbnfCompiler{named: named, rules: make([][][]int, len(names)), references: make(map[string]bool)}
}

// accountProductions charges newly built productions against the running limit
// and fails the moment it is crossed, so an adversarial grammar cannot allocate
// far past the cap before a single end-of-compile check fires.
func (c *gbnfCompiler) accountProductions(n int) error {
	c.productionCount += n
	if c.productionCount > gbnfMaxProductions {
		return fmt.Errorf("compiled grammar exceeds %d productions, limit is %d", c.productionCount, gbnfMaxProductions)
	}
	return nil
}

func (c *gbnfCompiler) compileNamed(name string, node *gbnfNode) error {
	id := c.named[name]
	productions, err := c.productionsFor(node)
	if err != nil {
		return err
	}
	c.rules[id] = productions
	return c.accountProductions(len(productions))
}

func (c *gbnfCompiler) productionsFor(node *gbnfNode) ([][]int, error) {
	if node.kind == gbnfNodeAlternate {
		out := make([][]int, 0, len(node.children))
		for _, child := range node.children {
			sequence, err := c.sequenceFor(child)
			if err != nil {
				return nil, err
			}
			out = append(out, sequence)
		}
		return out, nil
	}
	sequence, err := c.sequenceFor(node)
	if err != nil {
		return nil, err
	}
	return [][]int{sequence}, nil
}

func (c *gbnfCompiler) sequenceFor(node *gbnfNode) ([]int, error) {
	if node.kind == gbnfNodeEmpty {
		return nil, nil
	}
	if node.kind == gbnfNodeSequence {
		out := make([]int, 0, len(node.children))
		for _, child := range node.children {
			symbol, err := c.symbolFor(child)
			if err != nil {
				return nil, err
			}
			out = append(out, symbol)
		}
		return out, nil
	}
	symbol, err := c.symbolFor(node)
	if err != nil {
		return nil, err
	}
	return []int{symbol}, nil
}

func (c *gbnfCompiler) symbolFor(node *gbnfNode) (int, error) {
	switch node.kind {
	case gbnfNodeTerminal:
		c.terminals = append(c.terminals, node.matcher)
		return -len(c.terminals), nil
	case gbnfNodeReference:
		c.references[node.name] = true
		if id, ok := c.named[node.name]; ok {
			return id, nil
		}
		return 0, fmt.Errorf("references undefined rule %q", node.name)
	case gbnfNodeEmpty, gbnfNodeSequence, gbnfNodeAlternate:
		return c.addGenerated(node)
	case gbnfNodeRepeat:
		return c.addRepeat(node)
	default:
		return 0, fmt.Errorf("unsupported grammar node %d", node.kind)
	}
}

func (c *gbnfCompiler) addGenerated(node *gbnfNode) (int, error) {
	id := len(c.rules)
	c.rules = append(c.rules, nil)
	productions, err := c.productionsFor(node)
	if err != nil {
		return 0, err
	}
	c.rules[id] = productions
	if err := c.accountProductions(len(productions)); err != nil {
		return 0, err
	}
	return id, nil
}

func (c *gbnfCompiler) addRepeat(node *gbnfNode) (int, error) {
	child, err := c.symbolFor(node.children[0])
	if err != nil {
		return 0, err
	}
	id := len(c.rules)
	c.rules = append(c.rules, nil)
	productions := make([][]int, 0)
	if node.max >= 0 {
		for count := node.min; count <= node.max; count++ {
			production := make([]int, count)
			for i := range production {
				production[i] = child
			}
			productions = append(productions, production)
		}
	} else {
		base := make([]int, node.min)
		for i := range base {
			base[i] = child
		}
		productions = append(productions, base)
		recursive := []int{child, id}
		productions = append(productions, recursive)
	}
	c.rules[id] = productions
	if err := c.accountProductions(len(productions)); err != nil {
		return 0, err
	}
	return id, nil
}

func (c *gbnfCompiler) validateReferences() error {
	for reference := range c.references {
		if _, ok := c.named[reference]; !ok {
			return fmt.Errorf("grammar references undefined rule %q", reference)
		}
	}
	return nil
}
