# Contributing to Paperboy

Thank you for considering contributing to Paperboy! This guide will help you get started with the contribution process.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template when creating a new issue
- Include detailed steps to reproduce the bug
- Describe the expected behavior and what actually happens
- Include system information (OS, Python version, etc.)

### Suggesting Features

- Check if the feature has already been suggested in the Issues section
- Use the feature request template when creating a new issue
- Describe the feature in detail and why it would be valuable
- If possible, provide examples of how the feature would work

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes with descriptive commit messages
6. Push to your branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## Development Environment

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/paperboy.git
   cd paperboy
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file:

   ```bash
   cp config/.env.example config/.env
   ```

5. Edit the `.env` file with your API keys and settings

### Docker Development

Alternatively, you can use Docker for development:

```bash
docker-compose up --build
```

## Coding Guidelines

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Add type hints
- Write tests for new features
- Keep functions small and focused on a single task
- Use meaningful variable and function names

## Testing

Run the test suite using pytest:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=src
```

## Documentation

- Update the README.md if your changes affect usage
- Document new features in the appropriate section
- Update the API documentation if you modify endpoints

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
