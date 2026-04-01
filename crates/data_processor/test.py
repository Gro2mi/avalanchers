import data_processor
import os
import datetime

path = data_processor.__file__

# Get last modification time
mtime = os.path.getmtime(path)
dt = datetime.datetime.fromtimestamp(mtime)

print(f"Module path: {path}")
print(f"Last modified: {dt.isoformat()}")

meta = data_processor.MetaGrid(1, 3, 5, 1, 3857, 13, 15, data_processor.DataType.F32, data_processor.Variable.Velocity, data_processor.Unit.MetersPerSecond)
print(meta)

file = data_processor.F32Data.load("../compute_core/slope_angle.bin")
print(file.data)

# data = data_processor.F32Data(
#     meta,
#     [1.0, 2.0, 3.0]
# )
# data.save("test_f32data.bin")


# data_loaded = data_processor.F32Data.load("test_f32data.bin")
# print(data)
# print(data_loaded)
# assert meta == meta2

# assert data_loaded.metadata == data.metadata
# assert data_loaded.data == data.data
