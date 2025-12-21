# GeoAI Serve - Claude Code Setup Summary

**Project**: Python Backend Service for Geospatial AI Models
**Tech Stack**: Python 3.11+, FastAPI, Pydantic, uvicorn
**AI Models**: moondream, dinov2, samgeo (from https://github.com/opengeos/geoai)

---

## ✅ Skills Configuration

### 1. **python** (General Python)
- **Type**: Domain skill
- **Enforcement**: Suggestion (non-blocking)
- **Priority**: Medium
- **Coverage**: Type hints, pytest, package management (uv), code style

### 2. **python-backend** (FastAPI Development)
- **Type**: Guardrail
- **Enforcement**: ⚠️ **BLOCKING** (requires skill for FastAPI work)
- **Priority**: Critical
- **Coverage**: FastAPI patterns, Pydantic, async/await, authentication, database, deployment

**Triggers**:
- Keywords: fastapi, api, endpoint, router, pydantic, async, authentication, etc.
- File patterns: `**/main.py`, `**/app.py`, `**/api/**/*.py`, `**/routers/**/*.py`
- Content patterns: `from fastapi import`, `@app.get`, `class.*BaseModel`, etc.

**Bypass Options**:
- Environment: `export SKIP_PYTHON_BACKEND_SKILL=1`
- File marker: `# @skip-backend-skill`
- Session: Auto-skips after first use

---

## ✅ Agents Configuration

### 1. **code-architecture-reviewer** ✨ Updated for GeoAI
- **Model**: Sonnet
- **Purpose**: Review Python/FastAPI code for production readiness
- **Focus Areas**:
  - FastAPI patterns and async programming
  - AI model integration (loading, inference, memory management)
  - Geospatial AI specifics (samgeo, moondream, dinov2)
  - Resource management and cleanup
  - Security, validation, performance
  - Docker deployment considerations
- **Output**: Saves reviews to `.claude/reviews/[feature-name]-review.md`

### 2. **plan-reviewer**
- **Model**: Opus
- **Purpose**: Review implementation plans before coding
- **Focus**: Identifies issues, missing considerations, better alternatives

### 3. **web-research-specialist**
- **Model**: Sonnet
- **Purpose**: Research technical problems and solutions
- **Focus**: GitHub issues, Stack Overflow, documentation, forums

---

## ✅ Hooks Configuration

### UserPromptSubmit Hook
- **File**: `.claude/hooks/user-prompt-submit.sh`
- **Purpose**: Auto-suggests skills based on prompt and file context
- **Features**:
  - Analyzes user prompts for keywords and intent patterns
  - Checks file paths and content for context
  - Suggests relevant skills with scoring system
  - **BLOCKS** when python-backend skill is required
  - Respects skip conditions and environment variables

**Configuration**:
```bash
# Enable/disable
export CLAUDE_SKILL_SUGGESTIONS=true  # default

# Debug mode
export CLAUDE_DEBUG=true

# Skip specific skills
export SKIP_PYTHON_BACKEND_SKILL=1
export SKIP_PYTHON_SKILL=1
```

---

## 📁 Directory Structure

```
.claude/
├── agents/
│   ├── code-architecture-reviewer.md    ✨ Updated for Python/FastAPI/GeoAI
│   ├── plan-reviewer.md                 ✅ Generic, works for any project
│   └── web-research-specialist.md       ✅ Generic, works for any project
├── hooks/
│   ├── user-prompt-submit.sh            ✅ Main hook script
│   ├── skill-suggester.py               ✅ Python analysis engine
│   ├── README.md                        📖 Hook documentation
│   └── SETUP.md                         📖 Setup instructions
├── skills/
│   ├── python/
│   │   └── SKILL.md                     ✅ Python best practices
│   ├── python-backend/
│   │   └── SKILL.md                     ✅ FastAPI comprehensive guide
│   └── skill-rules.json                 ✅ Trigger rules (hybrid enforcement)
├── references/                          📚 API References
│   ├── geoai-models-api-reference.md    📖 Complete API docs for all 3 models
│   └── fastapi-endpoint-patterns.md     📖 FastAPI implementation patterns
├── reviews/                             📁 Code review outputs (created)
└── SETUP_SUMMARY.md                     📄 This file
```

---

## 🎯 Ready for Development

### What Works Now:

✅ **Skill Suggestions**: Automatically suggests `python` or `python-backend` based on your prompts
✅ **Guardrails**: Blocks FastAPI development until skill is loaded (can be bypassed)
✅ **Code Review**: Agent configured for Python/FastAPI/GeoAI code review
✅ **Plan Review**: Agent ready to review implementation plans
✅ **Research**: Agent ready to research technical problems

### Activation (Optional):

To enable the UserPromptSubmit hook in Claude Code, add to settings:

**Global**: `~/.config/claude-code/settings.json`
**Project**: `.claude/settings.json`

```json
{
  "hooks": {
    "userPromptSubmit": ".claude/hooks/user-prompt-submit.sh"
  }
}
```

---

## 🚀 Next Steps

1. ✅ Skills, agents, and hooks are configured
2. ⏭️ Start planning your GeoAI backend architecture
3. ⏭️ Begin implementing FastAPI endpoints
4. ⏭️ Integrate AI models (moondream, dinov2, samgeo)
5. ⏭️ Use code-architecture-reviewer agent after writing code
6. ⏭️ Containerize with Docker

---

## 📝 Example Workflow

**1. Start a new feature:**
```
User: "I want to create an API endpoint for SAM model inference"
Claude: [python-backend skill suggested/required]
User: /python-backend
Claude: [Loaded skill, ready to implement]
```

**2. After implementation:**
```
User: "Review the SAM endpoint code"
Claude: [Launches code-architecture-reviewer agent]
Agent: [Saves review to .claude/reviews/sam-endpoint-review.md]
Claude: [Reports findings, waits for approval before fixes]
```

**3. Before major work:**
```
User: "Review my plan to integrate dinov2 embeddings"
Claude: [Launches plan-reviewer agent]
Agent: [Identifies issues, suggests improvements]
```

---

**Version**: 1.0
**Last Updated**: 2025-12-21
**Status**: ✅ Ready for Development
