# Agent Guidelines for xmsr

## Build and Development Commands

### Building
- `uv build` - Build the package distribution files
- `uv sync --dev` - Install package with development dependencies

### Linting and Formatting
- `ruff check` - Check code quality, style, and potential issues
- `ruff format` - Automatically format code to match style guidelines
- `ruff check --fix` - Automatically fix fixable linting issues

### Testing
- No formal test suite configured yet
- For running individual tests: `python -m pytest tests/test_file.py::TestClass::test_method`

### Type Checking
- `mypy src/` - Check type annotations (if mypy is configured)

## Code Style Guidelines

### Imports
- Standard library imports first, grouped by category
- Third-party imports second (xarray, numpy, zarr, etc.)
- Local imports last, with blank line separation between groups
- Use specific imports: `from pathlib import Path` instead of `import os.path`
- Use relative imports for internal modules: `from .measurement import Measurement`

### Naming Conventions
- Functions and variables: `snake_case` (e.g., `current_index`, `param_coords`)
- Classes: `PascalCase` (e.g., `Measurement`, `VariableData`)
- Constants: `UPPER_CASE` (e.g., `_CURRENT_INDEX_KEY`)
- Private attributes/methods: `_leading_underscore` (e.g., `_current_index`)

### Type Hints
- Use comprehensive type hints for all public function parameters and return values
- Use `Optional[T]` or `T | None` for nullable types
- Use `Union[A, B]` or `A | B` for multiple possible types
- Use `TypedDict` for complex dictionary structures (e.g., `ProgressDict`)
- Use `dataclasses` for simple data containers with type annotations
- Specify collection types: `list[str]`, `dict[str, Any]`, `tuple[int, ...]`

### Error Handling and Logging
- Use logging extensively with appropriate levels:
  - `self.LOG.debug()` for detailed debugging information
  - `self.LOG.info()` for general operational information
  - `self.LOG.warning()` for warnings that don't stop execution
  - `self.LOG.error()` for errors that indicate problems
- Avoid bare `print()` statements - use logging instead
- Configure loggers per class: `self.LOG = logging.getLogger(self.__class__.__name__)`

### Documentation
- Use Google-style docstrings for all public methods and classes
- Document parameters with types: `Args: param_name (type): description`
- Document return values: `Returns: (type): description`
- Document exceptions: `Raises: ExceptionType: description`
- Include type information in docstrings when it adds clarity

### File Operations
- Use `pathlib.Path` instead of string-based path operations
- Prefer context managers for file operations
- Handle file permissions and existence checks appropriately
- Use absolute paths when necessary, but prefer relative paths for portability

### Data Structures
- Use `@dataclass` for simple data containers (e.g., `VariableData`)
- Use `TypedDict` for complex configuration dictionaries (e.g., `ProgressDict`)
- Leverage xarray `DataArray`/`Dataset` for scientific data storage and manipulation
- Use numpy arrays for numerical data with appropriate dtypes

### Code Organization
- Keep functions focused on single responsibilities
- Use descriptive variable names that indicate purpose
- Follow DRY principle (Don't Repeat Yourself)
- Use list/dict/set comprehensions for simple transformations
- Break complex logic into smaller, well-named helper methods

### Security and Best Practices
- Never log sensitive information (passwords, API keys, etc.)
- Use secure random generation when needed for cryptographic operations
- Validate inputs and handle edge cases gracefully
- Follow principle of least privilege in file operations
- Use environment variables for configuration when appropriate

## Development Workflow

1. Make changes to code following these guidelines
2. Run `ruff format` to ensure consistent formatting
3. Run `ruff check` to identify and fix code quality issues
4. Test changes manually or create appropriate test cases
5. Run `uv build` to ensure the package builds correctly
6. Commit changes with descriptive messages following conventional commits

## Dependencies

### Core Dependencies
- `numpy>=1.25` - Numerical computing and array operations
- `xarray<2025.5.0` - N-dimensional labeled arrays and datasets
- `zarr<3` - Chunked, compressed storage for array data
- `matplotlib>=3.10.6` - Plotting and visualization
- `tqdm>=4.67.1` - Progress bars for long-running operations

### Development Dependencies
- `ruff>=0.13.3` - Fast Python linter and code formatter
- `ipykernel>=6.30.1` - Jupyter kernel for interactive development

## Project Structure
- `src/xmsr/` - Main package code
- `examples/` - Usage examples and tutorials
- `tmp/` - Temporary measurement data storage
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Dependency lock file

## Measurement-Specific Guidelines

### Class Design
- Subclass `Measurement` for new measurement types
- Define `param_coords` as class attribute for parameter sweeps
- Implement `measure()` method for data acquisition
- Use `VariableData` for multi-variable measurements
- Override `prepare()` and `finish()` for setup/cleanup

### Data Storage
- Use Zarr format for efficient, chunked data storage
- Include comprehensive metadata in archives
- Leverage xarray's coordinate system for self-documenting data
- Support resume functionality for interrupted measurements

### Threading
- Use class-level `Event` objects for thread coordination
- Implement non-blocking execution with proper synchronization
- Handle thread safety when using shared resources
- Provide pause/resume capabilities via event signaling

### Configuration
- Use class attributes for default configuration
- Allow constructor parameters for customization
- Support both programmatic and declarative configuration
- Include timestamp and coordinate information in filenames by default