#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>

using namespace std;

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

int computeCapabilityToMMAVersion(int computeCapability) {
  if (computeCapability < 70) {
    return 0;
  } else if (computeCapability < 80) {
    return 1;
  } else if (computeCapability < 90) {
    return 2;
  } else if (computeCapability < 100) {
    // FIXME: temporarily add this to pass unis tests
    return 2;
  } else {
    assert(false && "computeCapability > 100 not supported");
    return 3;
  }
}

SmallVector<int64_t, 2> mmaVersionToShapePerWarp(int version) {
  if (version == 1)
    return {16, 16};
  else if (version == 2)
    return {16, 8};
  else {
    assert(false && "version not supported");
    return {0, 0};
  }
}
SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<triton::DotOp>(op);
      }) != slices.end())
    return {(unsigned)numWarps, 1};
  std::cout << "starting warpsPerTileV2" << std::endl;
  SmallVector<unsigned, 2> ret = {1, 1};
//   SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  bool changed = false;
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
//   do {
//     changed = false;
//     if (ret[0] * ret[1] >= numWarps)
//       break;
// if (shape[0] / shapePerWarp[0] / ret[0] >=
//         shape[1] / (shapePerWarp[1] * 2) / ret[1] &&
//     shape[1] / (shapePerWarp[1] * 2) / ret[1] >=
//         shape[2] / (shapePerWarp[2] * 4) / ret[2]) {
//   if (ret[0] < shape[0] / shapePerWarp[0]) {
//     ret[0] *= 2;
//   } else if (ret[1] < shape[1] / (shapePerWarp[1] * 2)) {
//     ret[1] *= 2;
//   } else {
//     ret[2] *= 2;
//   }
// } else if (shape[1] / (shapePerWarp[1] * 2) / ret[1] >=
//                shape[2] / (shapePerWarp[2] * 4) / ret[2]) {
//   if (ret[1] < shape[1] / (shapePerWarp[1] * 2)) {
//     ret[1] *= 2;
//   } else {
//     ret[2] *= 2;
//   }
// } else {
//   ret[2] *= 2;
// }
//   } while(true);


  std::cout << "warpsPerTile" << std::endl;
  std::cout << ret[0] << " " <<ret[1]<<std::endl;
  return ret;
}
// SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
//                                         const ArrayRef<int64_t> shape,
//                                         int numWarps) {
//   SetVector<Operation *> slices;
//   mlir::getForwardSlice(dotOp.getResult(), &slices);
//   if (llvm::find_if(slices, [](Operation *op) {
//         return isa<triton::DotOp>(op);
//       }) != slices.end())
//     return {(unsigned)numWarps, 1};

//   SmallVector<unsigned, 2> ret = {1, 1};
//   SmallVector<int64_t, 2> shapePerWarp = {16, 8};
//   // SmallVector<int64_t, 2> shapePerWarp = {32, 4};
//   bool changed = false;
//   // TODO (@daadaada): double-check.
//   // original logic in
//   // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L252
//   // seems buggy for shape = [32, 16] ?
//   do {
//     changed = false;
//     if (ret[0] * ret[1] >= numWarps)
//       break;
//     if (shape[0] / shapePerWarp[0] / ret[0] >=
//         shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
//       if (ret[0] < shape[0] / shapePerWarp[0]) {
//         ret[0] *= 2;
//       } else
//         ret[1] *= 2;
//     } else {
//       ret[1] *= 2;
//     }
//   } while (true);
//   std::cout << ret[0] << " " <<ret[1]<< std::endl;
//   return ret;
// }

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<triton::gpu::MmaEncodingAttr>())
      return failure();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();

    triton::gpu::MmaEncodingAttr mmaEnc;
    if (versionMajor == 1) {
      SetVector<Operation *> aBwdSlices, bBwdSlices;
      auto isCvt = [](Operation *op) { return isa<ConvertLayoutOp>(op); };
      getBackwardSlice(a, &aBwdSlices, isCvt);
      getBackwardSlice(b, &bBwdSlices, isCvt);
      // get the source of the first conversion found in slices
      auto getCvtArgOrder = [](Operation *op) {
        return cast<ConvertLayoutOp>(op)
            .getOperand()
            .getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .cast<BlockedEncodingAttr>()
            .getOrder();
      };
      bool isARow = true;
      bool isBRow = true;
      Operation *aOp = a.getDefiningOp();
      Operation *bOp = b.getDefiningOp();
      if (!aBwdSlices.empty())
        aOp = aBwdSlices[0];
      if (!bBwdSlices.empty())
        bOp = bBwdSlices[0];
      if (aOp)
        isARow = getCvtArgOrder(aOp)[0] == 1;
      if (bOp)
        isBRow = getCvtArgOrder(bOp)[0] == 1;

      mmaEnc = triton::gpu::MmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, numWarps, oldAType.getShape(),
          oldBType.getShape(), retShape, isARow, isBRow, mmaV1Counter++);
    } else if (versionMajor == 2) {

      // std::cout << "+++++++++++++" << std::endl;
      auto warpsPerTile = warpsPerTileV2(dotOp, retShape, numWarps);
      // SmallVector<unsigned, 2>warpsPerTile = {1, 4};
      // SmallVector<unsigned, 2>warpsPerTile = {4, 1};
      // SmallVector<unsigned, 2>warpsPerTile = {2, 1};
      // SmallVector<unsigned, 2>warpsPerTile = {1, 2};

      mmaEnc = triton::gpu::MmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, 0 /*versionMinor*/,
          warpsPerTile);
    } else {
      llvm_unreachable("Mma layout only supports versionMajor in {1, 2}");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    auto oldAOrder = oldAType.getEncoding()
                         .cast<triton::gpu::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<triton::gpu::BlockedEncodingAttr>()
                         .getOrder();
    auto oldBOrder = oldBType.getEncoding()
                         .cast<triton::gpu::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<triton::gpu::BlockedEncodingAttr>()
                         .getOrder();

    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        oldAType.getElementType());
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(),
        oldBType.getElementType());

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getAllowTF32());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPassCustom
    : public TritonGPUAccelerateMatmulBase<TritonGPUAccelerateMatmulPassCustom> {
public:
  TritonGPUAccelerateMatmulPassCustom() = default;
  TritonGPUAccelerateMatmulPassCustom(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::BlockedToMMA>(context, computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUAccelerateMatmulPassCustom(int computeCapability) {
  return std::make_unique<TritonGPUAccelerateMatmulPassCustom>(computeCapability);
}
