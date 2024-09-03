# build a tensorrt inference code
 
import tensorrt as trt
import os
from calibrator import Calibrator, CalibDataLoader

LOGGER = trt.Logger(trt.Logger.VERBOSE)


def buildEngine(
    onnx_file, engine_file, FP16_mode, INT8_mode, data_loader, calibration_table_path
):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if FP16_mode == True:
        config.set_flag(trt.BuilderFlag.FP16)

    elif INT8_mode == True:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(data_loader, calibration_table_path)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")

    with open(engine_file, "wb") as f:
        f.write(engine)


def main():
    
    onnx_file = "./models/mobilev2_model.onnx"
    engine_file = "./trt/mobilev2_model_int8.engine"
    calibration_cache = "./trt/mobilev2_model_calib.cache"

    FP16_mode = False
    INT8_mode = True

    dataloader = CalibDataLoader(batch_size=1, calib_count=1000)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print(
        "Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file)
    )
    buildEngine(
        onnx_file, engine_file, FP16_mode, INT8_mode, dataloader, calibration_cache
    )
    print("Export ENGINE success, Save as: ", engine_file)


if __name__ == "__main__":
    main()
