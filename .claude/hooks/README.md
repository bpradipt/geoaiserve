# Claude Code Hooks - Skill Suggestion System

This directory contains hooks for Claude Code that automatically suggest relevant skills based on your prompts and file context.

## Files

- `user-prompt-submit.sh` - Main hook script that runs before each prompt
- `skill-suggester.py` - Python script that analyzes prompts and suggests skills
- `README.md` - This file

## How It Works

When you submit a prompt to Claude Code, the `user-prompt-submit.sh` hook:

1. **Analyzes your prompt** for keywords and intent patterns
2. **Checks file context** (open files, file content, project structure)
3. **Matches against rules** defined in `../skills/skill-rules.json`
4. **Suggests relevant skills** if there's a strong match

## Configuration

### Enable/Disable Suggestions

```bash
# Disable skill suggestions
export CLAUDE_SKILL_SUGGESTIONS=false

# Re-enable (default)
export CLAUDE_SKILL_SUGGESTIONS=true
```

### Skip Specific Skills

You can skip suggestions for specific skills by setting environment variables:

```bash
# Skip python-backend skill suggestions
export SKIP_PYTHON_BACKEND_SKILL=1

# Skip python skill suggestions
export SKIP_PYTHON_SKILL=1
```

### File-Level Skip Markers

Add these comments to your Python files to skip skill validation:

```python
# @skip-skill-validation
# or
# @skip-backend-skill
# or
# @skip-python-skill
```

## Skill Rules Configuration

Edit `../skills/skill-rules.json` to customize when skills are suggested:

### Rule Structure

```json
{
  "skills": {
    "skill-name": {
      "enforcement": "suggestion",  // or "block", "warn"
      "priority": "high",           // critical, high, medium, low
      "description": "...",
      "promptTriggers": {
        "keywords": [...],          // Match these keywords
        "intentPatterns": [...]     // Match these regex patterns
      },
      "fileTriggers": {
        "pathPatterns": [...],      // Match file paths
        "contentPatterns": [...]    // Match file content
      },
      "suggestionMessage": "...",
      "skipConditions": {
        "sessionSkillUsed": true,   // Skip after first use
        "fileMarkers": [...],       // File comments to skip
        "envOverride": "ENV_VAR"    // Environment variable to skip
      }
    }
  }
}
```

### Enforcement Modes

- **suggestion** - Shows a suggestion but doesn't block (default)
- **warn** - Shows a warning but allows proceeding
- **block** - Requires the skill to be used (guardrail mode)

### Priority Levels

- **critical** - Always trigger when matched (score +100)
- **high** - Important, trigger for most matches (score +50)
- **medium** - Moderate, trigger for clear matches (score +25)
- **low** - Optional, trigger only for explicit matches (score +10)

## Example Use Cases

### Scenario 1: Creating a FastAPI Endpoint

**User Prompt:** "Create a new API endpoint for user authentication"

**Suggestion:**
```
💡 Suggestion: python-backend
   Modern Python API development with FastAPI...

   Use: /invoke python-backend
```

### Scenario 2: Working on Python Tests

**User Prompt:** "Add unit tests for the user service"

**Suggestion:**
```
💡 Suggestion: python
   Python coding standards, best practices...

   Use: /invoke python
```

### Scenario 3: Multiple Matches

If both `python` and `python-backend` match, the higher-scoring skill (based on specificity and priority) is suggested first.

## Debugging

Enable debug mode to see what the hook is doing:

```bash
export CLAUDE_DEBUG=true
```

Debug messages will appear in stderr showing:
- Prompt analysis
- File detection
- Skill matching logic
- Scores and reasons

## Testing the Hook

You can test the skill suggester directly:

```bash
# Test with a prompt only
python3 skill-suggester.py "Create a FastAPI endpoint"

# Test with a prompt and file context
python3 skill-suggester.py "Add authentication" app.py main.py

# Test with environment variable
SKIP_PYTHON_BACKEND_SKILL=1 python3 skill-suggester.py "Create API"
```

## Advanced Customization

### Adding Custom Keywords

Edit `skill-rules.json` and add keywords to the `keywords` array:

```json
"keywords": [
  "fastapi",
  "your-custom-keyword",
  "another-keyword"
]
```

### Adding Intent Patterns

Use regex patterns to match user intentions:

```json
"intentPatterns": [
  "(create|build).*?api",
  "implement.*?authentication",
  "your.*?custom.*?pattern"
]
```

### File Path Patterns

Use glob-style patterns to match files:

```json
"pathPatterns": [
  "**/api/**/*.py",
  "**/routers/**/*.py",
  "app.py",
  "main.py"
]
```

### Content Patterns

Use regex to match file content:

```json
"contentPatterns": [
  "from fastapi import",
  "@app\\.(get|post|put)",
  "class.*BaseModel"
]
```

## Best Practices

1. **Start with suggestions** - Use `enforcement: "suggestion"` for new skills
2. **Use specific patterns** - More specific patterns = better suggestions
3. **Test thoroughly** - Test your rules with various prompts
4. **Iterate** - Refine patterns based on actual usage
5. **Document** - Add clear descriptions and messages
6. **Respect user preferences** - Honor skip conditions

## Troubleshooting

### Hook not running?

- Check that the hook is executable: `ls -l user-prompt-submit.sh`
- Check that Python 3 is available: `which python3`
- Enable debug mode: `export CLAUDE_DEBUG=true`

### No suggestions appearing?

- Check that `skill-rules.json` exists and is valid JSON
- Verify your prompt matches keywords or patterns
- Lower the score threshold in `skill-suggester.py` (line 272)
- Enable debug mode to see matching logic

### Too many suggestions?

- Increase the score threshold
- Make patterns more specific
- Use skip conditions more aggressively
- Disable suggestions: `export CLAUDE_SKILL_SUGGESTIONS=false`

## Contributing

Feel free to customize this system for your needs:

1. Add new skills to `skill-rules.json`
2. Enhance the scoring algorithm in `skill-suggester.py`
3. Add new trigger types (e.g., git branch patterns, environment detection)
4. Improve the suggestion formatting

## Version

- Skill Rules: 1.1
- Hook System: 1.0
- Last Updated: December 2025
