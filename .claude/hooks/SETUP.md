# Hook Setup Guide

This guide helps you configure the UserPromptSubmit hook in Claude Code.

## Quick Setup

The hook files are already created and ready to use. You just need to configure Claude Code to use them.

## Option 1: Using Claude Code CLI Settings

Add this to your Claude Code settings file (`~/.config/claude-code/settings.json` or project-specific `.claude/settings.json`):

```json
{
  "hooks": {
    "userPromptSubmit": ".claude/hooks/user-prompt-submit.sh"
  }
}
```

## Option 2: Using Environment Variables

Set the hook path as an environment variable:

```bash
export CLAUDE_HOOK_USER_PROMPT_SUBMIT=".claude/hooks/user-prompt-submit.sh"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make it permanent.

## Option 3: Project-Specific Configuration

Create a `.claude/settings.json` file in your project root:

```bash
cat > .claude/settings.json << 'EOF'
{
  "hooks": {
    "userPromptSubmit": ".claude/hooks/user-prompt-submit.sh"
  },
  "skills": {
    "autoSuggest": true
  }
}
EOF
```

## Verification

Test that the hook is working:

### 1. Check Hook File

```bash
# Verify the hook exists and is executable
ls -l .claude/hooks/user-prompt-submit.sh

# Should show: -rwxr-xr-x ... user-prompt-submit.sh
```

### 2. Test the Skill Suggester Directly

```bash
# Test with a FastAPI-related prompt
python3 .claude/hooks/skill-suggester.py "Create a new FastAPI endpoint for user authentication"

# Expected output: Should suggest python-backend skill
```

### 3. Test with File Context

```bash
# Create a test file
echo "from fastapi import FastAPI" > test_api.py

# Test with file context
python3 .claude/hooks/skill-suggester.py "Add authentication" test_api.py

# Clean up
rm test_api.py
```

### 4. Test the Full Hook

```bash
# Set environment variables
export USER_PROMPT="Create a FastAPI endpoint"
export WORKING_DIRECTORY="$(pwd)"

# Run the hook
.claude/hooks/user-prompt-submit.sh

# Should display skill suggestions
```

## Customization

### Adjust Suggestion Threshold

Edit `.claude/hooks/skill-suggester.py` line 272:

```python
# Change this value to adjust when suggestions appear
if top_suggestion["score"] >= 30:  # Lower = more suggestions, Higher = fewer
```

### Add Custom Skills

1. Add your skill to `.claude/skills/`
2. Add rules to `.claude/skills/skill-rules.json`
3. Test with the skill suggester

### Modify Suggestion Format

Edit the `format_suggestion` method in `skill-suggester.py` to change how suggestions look.

## Troubleshooting

### Hook Not Firing

**Check Claude Code Configuration:**
```bash
# View current settings
cat ~/.config/claude-code/settings.json

# Or project settings
cat .claude/settings.json
```

**Verify Hook Path:**
The path should be relative to your project root or absolute.

### Python Not Found

```bash
# Check Python installation
which python3

# If missing, install Python 3:
# macOS: brew install python3
# Ubuntu: sudo apt install python3
# Windows: Download from python.org
```

### Permission Denied

```bash
# Make hooks executable
chmod +x .claude/hooks/*.sh .claude/hooks/*.py
```

### JSON Parse Errors

```bash
# Validate skill-rules.json
python3 -m json.tool .claude/skills/skill-rules.json

# Should output formatted JSON with no errors
```

## Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLAUDE_SKILL_SUGGESTIONS` | Enable/disable suggestions | `true` |
| `CLAUDE_DEBUG` | Enable debug logging | `false` |
| `SKIP_PYTHON_BACKEND_SKILL` | Skip python-backend suggestions | unset |
| `SKIP_PYTHON_SKILL` | Skip python suggestions | unset |
| `USER_PROMPT` | Current user prompt (set by Claude) | - |
| `WORKING_DIRECTORY` | Current working directory (set by Claude) | - |
| `OPEN_FILES` | Currently open files (set by Claude) | - |

## Advanced: Hook Chaining

If you have multiple hooks, you can chain them:

```bash
#!/usr/bin/env bash
# .claude/hooks/user-prompt-submit-main.sh

# Run skill suggester
.claude/hooks/user-prompt-submit.sh

# Run other custom hooks
.claude/hooks/your-other-hook.sh

exit 0
```

Then configure:
```json
{
  "hooks": {
    "userPromptSubmit": ".claude/hooks/user-prompt-submit-main.sh"
  }
}
```

## Next Steps

1. ✅ Verify hook is executable
2. ✅ Test skill suggester
3. ✅ Configure Claude Code settings
4. ✅ Try prompts and observe suggestions
5. ✅ Customize rules in skill-rules.json
6. ✅ Adjust thresholds and patterns as needed

## Examples of Working Prompts

These prompts should trigger skill suggestions:

**python-backend skill:**
- "Create a FastAPI endpoint for user login"
- "Add authentication to my API"
- "Implement JWT token validation"
- "Set up database connection with SQLAlchemy"

**python skill:**
- "Write a Python function with type hints"
- "Add unit tests using pytest"
- "Create a Python class for data validation"
- "Install dependencies using uv"

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Enable debug mode: `export CLAUDE_DEBUG=true`
3. Review hook logs in stderr
4. Validate JSON files
5. Test components individually

## Version History

- v1.0 (Dec 2025) - Initial release with python and python-backend skills
