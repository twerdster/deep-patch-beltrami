"""Export the original Deep Patch-Beltrami tensors as typed Core ML programs.

Run in a task environment with torch==2.5.1, coremltools==8.3.0, numpy<2,
h5py and Pillow, with this checkout installed or its src on PYTHONPATH.
The float32 model preserves the original graph. The mixed16 variant casts
only the expensive 80-to-80 spatial convolutions to float16; metric products,
determinants, inverse square roots and residual updates remain float32.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import torch
import coremltools as ct
from deep_patch_beltrami import assets
from deep_patch_beltrami.inference import DeepBeltrami
from deep_patch_beltrami.reproduce import OFFSET_FACTORS
from coreml_boundary import MaskedBoundaryNetwork, pack_rectangle


class CoreMLNetwork(torch.nn.Module):
    def __init__(self, models, sigma):
        super().__init__()
        self.source = DeepBeltrami(models, sigma, "cpu")

    def forward(self, noisy):
        output = noisy
        for graph in self.source.graphs:
            output = graph(output)
        return output


def heavy_spatial_conv(op):
    if op.op_type != "conv":
        return False
    shape = tuple(op.weight.shape)
    return len(shape)==4 and shape[:2]==(80,80) and shape[2:]==(7,7)


def path_hash(path):
    path = Path(path)
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    else:
        for item in sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith("._")):
            h.update(str(item.relative_to(path)).encode()+b"\0")
            h.update(item.read_bytes())
    return h.hexdigest()


def fixture(width,height,sigma,seed=1701):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:height,:width]
    clean=(110+35*np.sin(x/9)+25*np.cos(y/13)+15*np.sin((x+y)/7)).astype(np.float32)
    noisy=(clean+rng.normal(0,sigma,clean.shape)).astype(np.float32)
    return noisy.T[None,None].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--compiled-dir",type=Path)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--asset-dir",type=Path)
    parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--sigmas",type=int,nargs="+",default=[25])
    parser.add_argument("--precision",nargs="+",choices=["fp32","mixed16"],default=["fp32"])
    parser.add_argument("--tile-size",type=int,default=256)
    parser.add_argument("--minimum-size",type=int,default=1)
    parser.add_argument("--fixed",action="store_true",help="Export a fixed-size tile instead of a bounded flexible rectangle")
    parser.add_argument("--masked-boundary",action="store_true",help="Use a fixed tile with exact per-stage boundaries for smaller rectangles")
    parser.add_argument("--skip-validation",action="store_true")
    args=parser.parse_args()
    if args.masked_boundary:args.fixed=True
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models=assets.models(args.cache_dir,args.asset_dir,args.model_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.compiled_dir:args.compiled_dir.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema=1,format="CoreML MLProgram",input="noisy",output="denoised",
                  native_axis_order="N,C,visual-width,visual-height",halo=45,receptive_field=91,
                  minimum_size=args.minimum_size,maximum_size=args.tile_size,default_size=args.tile_size,
                  flexible=not args.fixed,models=[],torch=torch.__version__,coremltools=ct.__version__)
    manifest["boundary_mode"]="masked_rectangle" if args.masked_boundary else "native_rectangle"
    manifest_path=args.output_dir/"manifest.json"
    if manifest_path.is_file():
        manifest["models"]=json.loads(manifest_path.read_text()).get("models",[])
    for sigma in args.sigmas:
        reference_network=CoreMLNetwork(models,sigma).eval()
        sample=torch.from_numpy(fixture(args.tile_size,args.tile_size,sigma))
        for precision in args.precision:
            network=MaskedBoundaryNetwork(models,sigma) if args.masked_boundary else reference_network
            trace_inputs=pack_rectangle(sample,args.tile_size) if args.masked_boundary else sample
            with torch.inference_mode():
                traced=torch.jit.trace(network,trace_inputs,check_trace=False)
            start=time.perf_counter()
            suffix=(f"_masked{args.tile_size}" if args.masked_boundary else f"_fixed{args.tile_size}") if args.fixed else ""
            name=f"DeepBeltrami_sigma{sigma}_{precision}"+suffix
            shape=(1,1,args.tile_size,args.tile_size) if args.fixed else (
                1,1,ct.RangeDim(args.minimum_size,args.tile_size,default=args.tile_size),
                ct.RangeDim(args.minimum_size,args.tile_size,default=args.tile_size))
            compute_precision={"fp32":ct.precision.FLOAT32,
                               "mixed16":ct.transform.FP16ComputePrecision(op_selector=heavy_spatial_conv)}[precision]
            inputs=[ct.TensorType(name="noisy",shape=shape,dtype=np.float32)]
            if args.masked_boundary:
                inputs += [ct.TensorType(name="valid_mask",shape=shape,dtype=np.float32),
                           ct.TensorType(name="idx_x",shape=(args.tile_size,),dtype=np.int32),
                           ct.TensorType(name="idx_y",shape=(args.tile_size,),dtype=np.int32)]
            converted=ct.convert(traced,convert_to="mlprogram",minimum_deployment_target=ct.target.iOS18,
                                 inputs=inputs,
                                 outputs=[ct.TensorType(name="denoised",dtype=np.float32)],
                                 compute_precision=compute_precision,compute_units=ct.ComputeUnit.CPU_ONLY)
            converted.author="Aaron Wetzler"
            converted.short_description="Original five-stage Deep Patch-Beltrami denoiser"
            converted.user_defined_metadata.update({"sigma":str(sigma),"precision":precision,"halo":"45",
                "native_axes":"N,C,W,H","input_units":"0..255; Gaussian input remains unclipped",
                "boundary_mode":manifest["boundary_mode"],
                "offset_external":str(sigma*OFFSET_FACTORS[sigma]),"checkpoint_steps":"40000",
                "postprocessing":"Caller adds grayscale offset before inference, subtracts it afterward, then clips; RGB offset is zero."})
            package=args.output_dir/(name+".mlpackage")
            if package.exists():
                def ignore_disappeared(function,path,error):
                    if not isinstance(error[1],FileNotFoundError):raise error[1]
                shutil.rmtree(package,onerror=ignore_disappeared)
            converted.save(str(package))
            record=dict(sigma=sigma,precision=precision,package=package.name,package_sha256=path_hash(package),
                        export_seconds=time.perf_counter()-start,validation=[])
            if args.compiled_dir:
                compiled=Path(ct.models.utils.compile_model(str(package)))
                destination=args.compiled_dir/(name+".mlmodelc")
                if destination.exists():shutil.rmtree(destination)
                shutil.copytree(compiled,destination)
                record.update(compiled=destination.name,compiled_sha256=path_hash(destination))
            if not args.skip_validation:
                geometries=([(1,17),(127,129),(args.tile_size,193),(args.tile_size,args.tile_size)] if args.masked_boundary else
                            [(args.tile_size,args.tile_size)] if args.fixed else [(31,47),(97,63),(args.tile_size,193)])
                geometries=[(w,h)for w,h in geometries if args.minimum_size<=min(w,h) and max(w,h)<=args.tile_size]
                for unit_name,unit in [("cpu",ct.ComputeUnit.CPU_ONLY),("cpuAndGPU",ct.ComputeUnit.CPU_AND_GPU)]:
                    native=ct.models.MLModel(str(package),compute_units=unit)
                    for width,height in geometries:
                        value=fixture(width,height,sigma)
                        with torch.inference_mode():reference=reference_network(torch.from_numpy(value)).numpy()
                        then=time.perf_counter()
                        features={"noisy":value}
                        if args.masked_boundary:
                            packed=pack_rectangle(torch.from_numpy(value),args.tile_size)
                            features={key:tensor.numpy() for key,tensor in zip(("noisy","valid_mask","idx_x","idx_y"),packed)}
                        output=native.predict(features)["denoised"]
                        if not np.isfinite(output).all():
                            raise FloatingPointError(f"Non-finite Core ML tile: {name}, {unit_name}, {width}x{height}")
                        if args.masked_boundary:output=output[:,:,:width,:height]
                        elapsed=(time.perf_counter()-then)*1000
                        finite=bool(np.isfinite(output).all())
                        if not finite:
                            raise FloatingPointError(f"Non-finite Core ML output: {name}, {unit_name}, {width}x{height}")
                        error=output.astype(float)-reference.astype(float)
                        result=dict(width=width,height=height,units=unit_name,finite=finite,
                                    max_abs=float(np.max(np.abs(error))),rms=float(np.sqrt(np.mean(error**2))),
                                    milliseconds=elapsed,input_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                        record["validation"].append(result)
                        print(json.dumps(dict(model=name,**result)),flush=True)
            manifest["models"]=[old for old in manifest["models"]
                                if (old["sigma"],old["precision"])!=(sigma,precision)]
            manifest["models"].append(record)
            (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print("Saved "+str(package),flush=True)
    print(args.output_dir/"manifest.json",flush=True)


if __name__=="__main__":main()
