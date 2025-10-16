# Contributing to Waste Classifier

First off, thank you for considering contributing to Waste Classifier! 🎉

It's people like you that make Waste Classifier such a great tool for the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Code Style](#python-code-style)
  - [Documentation](#documentation)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include as many details as possible:

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Windows 10, Ubuntu 22.04]
 - Python Version: [e.g. 3.10.5]
 - TensorFlow Version: [e.g. 2.13.0]
 - GPU: [e.g. NVIDIA RTX 3060, or "None"]

**Additional context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Clear title and description** of the enhancement
- **Use case**: Explain why this would be useful
- **Possible implementation**: If you have ideas on how to implement it
- **Examples**: If applicable, provide examples from other projects

**Enhancement categories:**

- 🚀 New features
- ⚡ Performance improvements
- 📚 Documentation improvements
- 🎨 UI/UX improvements
- 🔧 Code quality improvements

### Your First Code Contribution

Unsure where to begin? You can start by looking through these issues:

- `good-first-issue` - Issues suitable for beginners
- `help-wanted` - Issues that need assistance

### Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our style guidelines

3. **Add tests** if applicable

4. **Update documentation** if needed

5. **Ensure all tests pass**:
   ```bash
   python -m pytest tests/
   ```

6. **Commit your changes** with a clear commit message

7. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

8. **Open a Pull Request** with a clear title and description

**Pull Request Template:**

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran.

## Checklist:
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
```

---

## Development Setup

### 1. Clone Your Fork

```bash
git clone https://github.com/AnHgPhamE/waste_classifier.git
cd waste_classifier
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install pytest black flake8 mypy
```

### 4. Prepare Development Data

```bash
# Download a small subset for testing
# Or use the full dataset
python src/data_prep.py
```

### 5. Verify Installation

```bash
# Run tests
python -m pytest tests/

# Check code style
black --check src/
flake8 src/
```

---

## Style Guidelines

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

**Good commit message examples:**

```
Add YOLOv8 integration for object detection

- Integrate ultralytics YOLOv8n model
- Add bounding box visualization
- Update preprocessing to use PIL for consistency
- Add tests for object detection pipeline

Fixes #123
```

**Commit message prefixes:**

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Formatting, missing semicolons, etc.
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Line length: 88 characters (Black default)
# Use Black for formatting
black src/

# Docstring format: Google style
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass

# Type hints are encouraged
def process_image(image_path: str) -> np.ndarray:
    """Process image and return array."""
    pass

# Use meaningful variable names
# Good:
user_confidence_threshold = 0.5

# Bad:
thresh = 0.5
```

**Import organization:**

```python
# 1. Standard library imports
import os
import sys
from datetime import datetime

# 2. Third-party imports
import numpy as np
import tensorflow as tf
from PIL import Image

# 3. Local imports
from config import IMG_SIZE, CLASS_NAMES
from utils import save_model
```

### Documentation

- **Docstrings**: All public functions, classes, and methods must have docstrings
- **Comments**: Use comments to explain "why", not "what"
- **README**: Update README.md if you change functionality
- **Inline documentation**: Keep it concise and relevant

```python
# Good comment (explains why)
# Use PIL instead of OpenCV to match training preprocessing
image = Image.open(path)

# Bad comment (states the obvious)
# Open the image
image = Image.open(path)
```

---

## Testing

### Running Tests

```bash

python -m pytest tests/

# Run specific test file
python -m pytest tests/test_predict.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

Place tests in the `tests/` directory:

```python

import pytest
import numpy as np
from src.predict import preprocess_image

def test_preprocess_image_shape():
    """Test that preprocessing returns correct shape."""
    result = preprocess_image("test_image.jpg")
    assert result.shape == (1, 224, 224, 3)

def test_preprocess_image_range():
    """Test that preprocessing normalizes to [0, 1]."""
    result = preprocess_image("test_image.jpg")
    assert result.min() >= 0.0
    assert result.max() <= 1.0
```

---

## Project Structure

```
src/
├── config.py           # Configuration constants
├── utils.py            # Utility functions
├── data_prep.py        # Data preparation
├── train.py            # Model training
├── predict.py          # Static prediction
├── predict_realtime.py # Real-time detection
└── evaluate.py         # Model evaluation

tests/
├── test_config.py
├── test_utils.py
├── test_preprocessing.py
└── test_model.py
```

**Key principles:**

- **Modularity**: Each file has a single, clear responsibility
- **Configuration**: All constants in `config.py`
- **DRY**: Don't Repeat Yourself - use utility functions
- **Error handling**: Graceful error messages and fallbacks

---

## Areas for Contribution

### High Priority

- [ ] **Improve model accuracy**: Experiment with different architectures
- [ ] **Add more tests**: Increase test coverage
- [ ] **Optimize performance**: Speed up inference
- [ ] **Better error messages**: More helpful user feedback

### Medium Priority

- [ ] **Web interface**: Flask/FastAPI web app
- [ ] **Mobile support**: TensorFlow Lite conversion
- [ ] **Docker container**: Containerized deployment
- [ ] **Cloud deployment**: AWS/GCP/Azure guides

### Low Priority

- [ ] **Additional visualizations**: Training curves, attention maps
- [ ] **Data augmentation**: More augmentation techniques
- [ ] **Model ensemble**: Combine multiple models
- [ ] **Multi-language support**: i18n for UI

---

## Recognition

Contributors will be recognized in:

- README.md Contributors section
- GitHub Contributors page
- Release notes for significant contributions

---

## Questions?

Feel free to:

- Open an issue with the `question` label
- Contact the maintainers directly
- Join our community discussions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🙏**

Every contribution, no matter how small, makes a difference. Together, we can build something amazing for a cleaner planet! 🌍♻️

