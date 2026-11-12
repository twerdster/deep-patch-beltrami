"""Command-line interface for Deep Patch-Beltrami inference and reproduction."""
from pathlib import Path
import argparse
import sys


def common(parser):
    parser.add_argument("--cache-dir", type=Path, help="Download/extraction cache (default: platform user cache)")
    parser.add_argument("--asset-dir", type=Path, help="Use verified release files in this directory without downloading them")
    parser.add_argument("--model-dir", type=Path, help="Existing extracted released model directory")


def runtime(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:INDEX")
    parser.add_argument("--threads", type=int, default=4)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deep-patch-beltrami", description="Released Deep Patch-Beltrami inference and thesis-table reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-models", help="Download and verify the original learned tensors")
    common(fetch)
    prepare = commands.add_parser("prepare-bsd68", help="Prepare public clean images and the original Gaussian noise")
    common(prepare)
    prepare.add_argument("--clean-dir", type=Path, help="Existing pinned BSD68 PNGs; otherwise download them separately")
    denoise = commands.add_parser("denoise", help="Denoise a grayscale or RGB image")
    common(denoise)
    runtime(denoise)
    denoise.add_argument("input", type=Path)
    denoise.add_argument("output", type=Path)
    denoise.add_argument("--sigma", type=int, required=True, choices=[5,10,15,20,25,30,40,50])
    denoise.add_argument("--offset-protocol", choices=["archived","none"], default="archived")
    reproduce = commands.add_parser("reproduce", help="Run the full thesis suite, fetching public BSD68 inputs as needed")
    common(reproduce)
    runtime(reproduce)
    reproduce.add_argument("dataset", choices=["bsd68","portraits100"])
    reproduce.add_argument("--clean-dir", type=Path, help="Existing pinned BSD68 PNGs")
    reproduce.add_argument("--suite", type=Path, help="User-supplied compatible HDF5 suite (required for portraits100)")
    reproduce.add_argument("--sigmas", type=int, nargs="+", default=[5,10,15,20,25,30,40,50])
    reproduce.add_argument("--limit", type=int, help="Smoke test only this many images; does not claim full-table reproduction")
    reproduce.add_argument("--offset-protocol", choices=["archived","none"], default="archived")
    reproduce.add_argument("--output", type=Path, default=Path("deep-reproduction.json"))
    args = parser.parse_args(argv)
    try:
        from . import assets
        model_dir = assets.models(args.cache_dir,args.asset_dir,args.model_dir)
        if args.command == "fetch-models":
            print(model_dir)
            return 0
        if args.command == "prepare-bsd68":
            from .reproduce import PublicBSD68, SIGMAS
            suite = PublicBSD68(args.cache_dir,args.asset_dir,args.clean_dir)
            try:
                for sigma in SIGMAS:
                    for name in suite.names:
                        suite.case(sigma,name)
                print("Verified all 544 original noisy inputs; model tensors and BSD68 are ready.")
            finally:
                suite.close()
            return 0
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if args.command == "denoise":
            import numpy as np
            from PIL import Image
            from .inference import DeepBeltrami, configure
            from .reproduce import OFFSET_FACTORS
            image = Image.open(args.input)
            gray = image.mode in ("1","L","I","I;16","F")
            array = np.asarray(image.convert("L" if gray else "RGB"),dtype=np.float32)
            native = array.T[None,None] if gray else array.transpose(2,1,0)[None]
            configure(device,args.threads)
            model = DeepBeltrami(model_dir,args.sigma,device)
            offset = args.sigma*OFFSET_FACTORS[args.sigma] if gray and args.offset_protocol=="archived" else 0
            result = model(native,offset)
            natural = result[0,0].T if gray else result[0].transpose(2,1,0)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.clip(natural,0,255).round().astype(np.uint8)).save(args.output)
            print(args.output)
            return 0
        from .reproduce import LocalSuite, PublicBSD68, evaluate
        if args.dataset == "portraits100" and args.suite is None:
            parser.error("portraits100 requires --suite; portrait images are not redistributed")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        suite = LocalSuite(args.suite) if args.suite else PublicBSD68(args.cache_dir,args.asset_dir,args.clean_dir)
        try:
            report = evaluate(suite,model_dir,args.dataset,device,args.sigmas,args.threads,
                              args.offset_protocol,args.limit,args.output)
        finally:
            suite.close()
        print(f"Saved {len(report['cases'])} finite cases to {args.output}")
        return 0
    except ImportError as error:
        parser.exit(1,"Install the runtime dependencies with: pip install -e .\n"+str(error)+"\n")
    except (OSError,ValueError,RuntimeError,ArithmeticError) as error:
        parser.exit(1,f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
