import inspect 

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]
        try:
            # with Profile() as dt:
            #     f, model = inner_func(*args, **kwargs)
            # LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            # return f, model
            print("in try export onnx")
        except Exception as e:
            # LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            print("export failed")
            return None, None

    return outer_func



@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    # YOLOv5 ONNX export
    # check_requirements("onnx>=1.12.0")
    import onnx

    # LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))

    # output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    # if dynamic:
    #     dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
    #     if isinstance(model, SegmentationModel):
    #         dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
    #         dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
    #     elif isinstance(model, DetectionModel):
    #         dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    # torch.onnx.export(
    #     model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
    #     im.cpu() if dynamic else im,
    #     f,
    #     verbose=False,
    #     opset_version=opset,
    #     do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
    #     input_names=["images"],
    #     output_names=output_names,
    #     dynamic_axes=dynamic or None,
    # )

    # # Checks
    # model_onnx = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model

    # # Metadata
    # d = {"stride": int(max(model.stride)), "names": model.names}
    # for k, v in d.items():
    #     meta = model_onnx.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(model_onnx, f)

    # Simplify
    # if simplify:
    #     try:
    #         cuda = torch.cuda.is_available()
    #         check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1"))
    #         import onnxsim

    #         LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
    #         model_onnx, check = onnxsim.simplify(model_onnx)
    #         assert check, "assert check failed"
    #         onnx.save(model_onnx, f)
    #     except Exception as e:
    #         LOGGER.info(f"{prefix} simplifier failure: {e}")
    # return f, model_onnx


