# python/test_bindings.py

import pytest
import os
import sys

# Add the parent directory to the sys.path if running tests directly
# This might be needed if `maturin develop` installs it in a way
# that Python's default path doesn't immediately find it, or for IDEs.
# However, `maturin develop` usually handles this correctly.
# If you put test_bindings.py directly in the project root, it's simpler.
# But for a cleaner structure, keep it in a 'python' or 'tests' subfolder.

# --- Crucial step: Import your Rust-built module ---
try:
    import data_processor
except ImportError:
    # If data_processor is not found, it might be due to PYTHONPATH or maturin develop not run
    print("Error: Could not import 'data_processor'.")
    print("Make sure you've run 'maturin develop' in your project root.")
    sys.exit(1)


# Define a temporary file path for serialization tests
TEMP_FILE_PATH = "temp_grid_data.bin"

# --- Test Functions ---

def test_data_type_enum():
    """Test the DataType enum creation and conversion."""
    f16_type = data_processor.DataType.F16
    f32_type = data_processor.DataType(32) # Using the #[new] constructor
    f64_type = data_processor.DataType.F64

    assert f16_type.as_int_py() == 16
    assert f32_type.as_int_py() == 32
    assert f64_type.as_int_py() == 64

    assert repr(f16_type) == "DataType.F16"
    assert repr(f32_type) == "DataType.F32"

    with pytest.raises(ValueError, match="Invalid DataType integer value"):
        data_processor.DataType(100)

def test_metadata_creation_and_access():
    """Test Metadata object creation and attribute access."""
    meta = data_processor.Metadata(width=10, height=20, data_type=data_processor.DataType.F32)

    assert meta.width == 10
    assert meta.height == 20
    assert meta.version == 1
    assert meta.data_type().as_int_py() == 32 # Accessing through the getter

    # Test setter
    meta.width = 100
    meta.height = 50
    meta.set_data_type(data_processor.DataType.F64)
    assert meta.width == 100
    assert meta.height == 50
    assert meta.data_type().as_int_py() == 64

    # Test magic bytes getter
    expected_magic_bytes = 0x47415641 # 'AVAG' in little-endian
    assert meta.magic_bytes() == expected_magic_bytes

def test_raw_data_creation_and_parsing_f32():
    """Test RawData creation, raw bytes, and parsing for F32."""
    width, height = 2, 2
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F32)
    original_data = [1.0, 2.5, 3.0, 4.5] # 2x2 = 4 elements
    raw_data_obj = data_processor.RawData(meta, original_data)

    assert raw_data_obj.metadata.width == width
    assert raw_data_obj.metadata.height == height
    assert raw_data_obj.metadata.data_type().as_int_py() == 32

    # Check raw bytes length
    # 4 elements * 4 bytes/f32 = 16 bytes
    assert len(raw_data_obj.raw_bytes()) == width * height * 4

    # Check parsed data
    parsed_data = raw_data_obj.parsed_data()
    assert parsed_data == original_data

def test_raw_data_creation_and_parsing_f16():
    """Test RawData creation, raw bytes, and parsing for F16 (with tolerance)."""
    width, height = 2, 2
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F16)
    original_data = [0.1, 1.2, 3.4, 5.6] # 2x2 = 4 elements
    raw_data_obj = data_processor.RawData(meta, original_data)

    assert raw_data_obj.metadata.data_type().as_int_py() == 16
    assert len(raw_data_obj.raw_bytes()) == width * height * 2 # 2 bytes/f16

    parsed_data = raw_data_obj.parsed_data()
    # F16 loses precision, so compare with tolerance
    for i in range(len(original_data)):
        assert abs(original_data[i] - parsed_data[i]) < 0.005 # Adjust tolerance if needed

def test_raw_data_creation_and_parsing_f64():
    """Test RawData creation, raw bytes, and parsing for F64."""
    width, height = 1, 3
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F64)
    original_data = [123.456, 789.012, 345.678]
    raw_data_obj = data_processor.RawData(meta, original_data)

    assert raw_data_obj.metadata.data_type().as_int_py() == 64
    assert len(raw_data_obj.raw_bytes()) == width * height * 8 # 8 bytes/f64

    parsed_data = raw_data_obj.parsed_data()
    assert parsed_data == original_data

@pytest.fixture(scope="module", autouse=True)
def cleanup_temp_file():
    """Fixture to ensure the temporary file is cleaned up after all tests in this module."""
    yield # Run tests
    if os.path.exists(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH)
        print(f"\nCleaned up {TEMP_FILE_PATH}")


def test_save_load_roundtrip_f32():
    """Test saving and loading RawData with F32 data."""
    width, height = 5, 5
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F32)
    original_data = [float(i) for i in range(width * height)]
    raw_data_original = data_processor.RawData(meta, original_data)

    serialized_bytes = data_processor.save_raw_data(raw_data_original)
    
    # Save to file
    with open(TEMP_FILE_PATH, "wb") as f:
        f.write(serialized_bytes)

    # Load from file
    with open(TEMP_FILE_PATH, "rb") as f:
        loaded_bytes = f.read()
    
    raw_data_loaded = data_processor.load_raw_data(loaded_bytes)

    # Assertions
    assert raw_data_loaded.metadata == raw_data_original.metadata
    assert raw_data_loaded.parsed_data() == raw_data_original.parsed_data()
    assert raw_data_loaded.raw_bytes() == raw_data_original.raw_bytes() # Check raw bytes too

def test_save_load_roundtrip_f16():
    """Test saving and loading RawData with F16 data (with tolerance)."""
    width, height = 3, 4
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F16)
    original_data = [float(i) * 0.0123 for i in range(width * height)]
    raw_data_original = data_processor.RawData(meta, original_data)

    serialized_bytes = data_processor.save_raw_data(raw_data_original)
    with open(TEMP_FILE_PATH, "wb") as f:
        f.write(serialized_bytes)
    with open(TEMP_FILE_PATH, "rb") as f:
        loaded_bytes = f.read()
    raw_data_loaded = data_processor.load_raw_data(loaded_bytes)

    assert raw_data_loaded.metadata == raw_data_original.metadata

    # For F16, compare parsed data with tolerance
    loaded_parsed_data = raw_data_loaded.parsed_data()
    original_parsed_data = raw_data_original.parsed_data() # This also converts original to F32 via Rust
    for i in range(len(original_data)):
        assert abs(original_parsed_data[i] - loaded_parsed_data[i]) < 0.005 # Tolerance

def test_save_load_roundtrip_f64():
    """Test saving and loading RawData with F64 data."""
    width, height = 7, 2
    meta = data_processor.Metadata(width=width, height=height, data_type=data_processor.DataType.F64)
    original_data = [float(i) * 987.654321 for i in range(width * height)]
    raw_data_original = data_processor.RawData(meta, original_data)

    serialized_bytes = data_processor.save_raw_data(raw_data_original)
    with open(TEMP_FILE_PATH, "wb") as f:
        f.write(serialized_bytes)
    with open(TEMP_FILE_PATH, "rb") as f:
        loaded_bytes = f.read()
    raw_data_loaded = data_processor.load_raw_data(loaded_bytes)

    assert raw_data_loaded.metadata == raw_data_original.metadata
    assert raw_data_loaded.parsed_data() == raw_data_original.parsed_data()
    assert raw_data_loaded.raw_bytes() == raw_data_original.raw_bytes()


# You can run these tests directly from this file for quick checks
if __name__ == "__main__":
    # If not using pytest, you'd call functions like this:
    # test_data_type_enum()
    # test_metadata_creation_and_access()
    # test_raw_data_creation_and_parsing_f32()
    # test_raw_data_creation_and_parsing_f16()
    # test_raw_data_creation_and_parsing_f64()
    # test_save_load_roundtrip_f32()
    # test_save_load_roundtrip_f16()
    # test_save_load_roundtrip_f64()

    # For proper pytest execution, ensure you have pytest installed: pip install pytest
    # Then navigate to your_project_root/python/ and run:
    # pytest test_bindings.py
    # Or from project root: pytest python/test_bindings.py

    print("Running tests with pytest...")
    # This will run pytest on the current file
    pytest.main([__file__])