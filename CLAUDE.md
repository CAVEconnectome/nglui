# NGLUI Development Guide

This guide documents the development workflow, testing strategies, and coding standards for the NGLUI package.

## Development Environment

### Package Management: UV + Poe

This project uses **UV** for dependency management and **Poe** for task running instead of the system Python environment:

```bash
# Install dependencies (automatically creates virtual environment)
uv sync

# Run tasks via Poe (preferred over direct python commands)
poe test          # Run tests with coverage
poe doc-preview   # Preview documentation
poe bump patch    # Bump version (patch/minor/major)
poe drybump patch # Dry run version bump
```

### Key Commands

From `pyproject.toml`:

- **Testing**: `poe test` → `uv run pytest --cov=nglui tests`
- **Documentation**: `poe doc-preview` → `uv run mkdocs serve`
- **Version Management**: `poe bump patch/minor/major`
- **Linting**: `uv run ruff check src/ tests/`

## Python Version Requirements

- **Minimum**: Python 3.10+ (leverages match statements and improved type annotations)
- **Tested**: Python 3.10, 3.11, 3.12
- **Compatibility**: Use `typing-extensions` for Python < 3.11 features

## Type Hinting Standards

### Required Practices

- **All functions/methods** must have complete type annotations
- **All class attributes** must have type hints (use `attrs` with type annotations)
- **Import patterns**:
  ```python
  from typing import Optional, Union, Literal, Any, Dict, List, Tuple
  from typing_extensions import Self  # For Python < 3.11
  ```

### Common Patterns

```python
from typing import Optional, Union, Literal
import attrs
import numpy as np
import pandas as pd

@attrs.define
class ExampleClass:
    # Required attributes with types
    name: str
    values: List[float] = attrs.field(factory=list)
    
    # Optional attributes
    description: Optional[str] = None
    
    # Use converters for type safety
    point: List[float] = attrs.field(converter=strip_numpy_types)
    
    # Use validators for constraints
    resolution: Optional[np.ndarray] = attrs.field(
        default=None, 
        converter=attrs.converters.optional(np.array)
    )

def process_data(
    data: pd.DataFrame,
    column: Union[str, List[str]],
    optional_param: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process dataframe with proper type annotations."""
    pass
```

## Testing Strategy

### Dual-Level Testing Approach

Always implement **both** high-level integration tests and low-level unit tests:

#### High-Level Integration Tests
Focus on real-world workflows and end-to-end functionality:

```python
def test_complete_annotation_workflow(self):
    """Test full annotation workflow with real data."""
    # Create realistic DataFrame
    df = pd.DataFrame({
        'x': [100, 200, 300],
        'y': [150, 250, 350], 
        'z': [10, 20, 30],
        'segment_id': [12345, 67890, 11111]
    })
    
    # Test complete workflow
    vs = ViewerState()
    vs.add_points(
        data=df,
        point_column=['x', 'y', 'z'],
        segment_column='segment_id'
    )
    
    # Verify end-to-end behavior
    assert len(vs.layers) == 1
    assert vs.layers[0].layer_type == 'annotation'
```

#### Low-Level Unit Tests
Focus on individual methods, edge cases, and error conditions:

```python
def test_scale_points_edge_cases(self):
    """Test scaling with edge cases."""
    point = PointAnnotation(point=[100, 200, 300])
    
    # Test zero scaling
    point._scale_points([0, 1, 2])
    assert point.point == [0.0, 200.0, 600.0]
    
    # Test negative scaling
    point = PointAnnotation(point=[100, 200, 300])
    point._scale_points([-1, -1, -1])
    assert point.point == [-100.0, -200.0, -300.0]
```

### Testing Guidelines

- **Coverage Target**: Aim for >90% line coverage, >85% branch coverage
- **Test Organization**: Mirror source structure in `tests/` directory
- **Mocking Strategy**: Mock external dependencies (neuroglancer), test actual behavior for internal logic
- **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple inputs
- **Fixtures**: Create reusable test data in `conftest.py`

### Running Tests

```bash
# Full test suite with coverage
poe test

# Specific test file
uv run pytest tests/test_ngl_annotations.py -v

# Coverage report
uv run pytest --cov=nglui --cov-report=html tests/
```

## Code Architecture Patterns

### Core Components

- **StateBuilder**: Main entry point for creating neuroglancer states
- **Annotations**: Point, Line, Ellipsoid, BoundingBox with neuroglancer conversion
- **Components**: ImageLayer, AnnotationLayer, SegmentationLayer
- **DataMap**: Dynamic data binding with priority system
- **Utils**: Type conversion, color parsing, coordinate handling

### Key Design Patterns

1. **Attrs Classes**: Use `@attrs.define` for all data classes
2. **Converter Functions**: `strip_numpy_types`, coordinate transformers
3. **Method Chaining**: Fluent API for building complex states
4. **DataMap Priority**: Higher numbers override lower numbers for dynamic updates
5. **Column Handling**: Support both string columns and list column specifications

### Point Column Feature

Support flexible coordinate specification:

```python
# Explicit column list
point_column=['x', 'y', 'z']

# Prefix expansion
point_column='position'  # → ['position_x', 'position_y', 'position_z']
```

## Common Development Workflows

### Adding New Annotation Types

1. Create class inheriting from `AnnotationBase`
2. Add type hints for all attributes
3. Implement `_scale_points()` method
4. Implement `to_neuroglancer()` method  
5. Add comprehensive tests (unit + integration)
6. Update documentation

### Adding New Layer Types

1. Create class inheriting from `LayerBase`
2. Implement required abstract methods
3. Add DataMap integration
4. Create builder methods in StateBuilder
5. Add comprehensive tests
6. Update documentation

### Bug Fixes

1. Write failing test first (TDD approach)
2. Implement minimal fix
3. Ensure all tests pass
4. Check type annotations are correct
5. Run full test suite: `poe test`

## Documentation Standards

- **Docstrings**: Use Google/NumPy style docstrings
- **Examples**: Include usage examples in docstrings
- **Type Information**: Document parameter and return types in docstrings
- **API Documentation**: Use mkdocs with mkdocstrings for auto-generation

```python
def add_points(
    self, 
    data: pd.DataFrame,
    point_column: Union[str, List[str]],
    segment_column: Optional[str] = None
) -> Self:
    """Add point annotations to the viewer state.
    
    Args:
        data: DataFrame containing point data
        point_column: Column name(s) for coordinates. Can be:
            - Single string for prefix (e.g., 'pos' → ['pos_x', 'pos_y', 'pos_z'])  
            - List of column names (e.g., ['x', 'y', 'z'])
        segment_column: Optional column containing segment IDs
        
    Returns:
        Self for method chaining
        
    Examples:
        >>> vs = ViewerState()
        >>> vs.add_points(df, point_column=['x', 'y', 'z'])
        >>> vs.add_points(df, point_column='position')  # Uses position_x, position_y, position_z
    """
```

## Pre-Commit Workflow

Before committing changes:

```bash
# Run full test suite
poe test

# Run linting
uv run ruff check src/ tests/

# Check type annotations
uv run mypy src/ --ignore-missing-imports  # if mypy is configured

# Ensure documentation builds
poe doc-preview
```

## Release Process

```bash
# Check what will be bumped
poe drybump patch  # or minor/major

# Bump version and create tag
poe bump patch     # or minor/major
```

This automatically:
- Updates version in `pyproject.toml` and `src/nglui/__init__.py`
- Creates git commit and tag
- Runs pre/post commit hooks including `uv sync`