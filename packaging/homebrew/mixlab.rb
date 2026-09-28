# Rendered from packaging/homebrew/mixlab.rb in https://github.com/mrothroc/mixlab
# at @RELEASE_TAG@ by the publish-homebrew workflow. Edits made in the tap are
# overwritten by the next release; change that source file instead.
class Mixlab < Formula
  desc "ML architecture exploration tool — JSON configs, Go IR, Metal/CUDA"
  homepage "https://github.com/mrothroc/mixlab"
  url "https://github.com/mrothroc/mixlab.git",
      tag:      "@RELEASE_TAG@",
      revision: "@RELEASE_REVISION@"
  license "MIT"
  head "https://github.com/mrothroc/mixlab.git", branch: "main"

  depends_on "go" => :build
  depends_on :macos
  depends_on "mlx"

  # Homebrew has no versioned mlx formula and depends_on takes no version
  # predicate, so the tested range is asserted here instead. MLX 0.32.1 changed
  # gather VJP semantics under a patch bump and silently broke MoE and bf16
  # training, so an untested MLX is treated as a hard error rather than allowed
  # through. Widen this after running the -tags mlx suite against the new MLX.
  # scripts/package_macos.py reads these two constants, so they stay the one
  # record of which MLX was tested.
  MLX_TESTED_MINIMUM = "0.32.0".freeze
  MLX_TESTED_BELOW = "0.33.0".freeze

  def install
    mlx_version = Formula["mlx"].version
    if mlx_version < Version.new(MLX_TESTED_MINIMUM) ||
       mlx_version >= Version.new(MLX_TESTED_BELOW)
      odie <<~EOS
        mixlab #{version} is tested against MLX >=#{MLX_TESTED_MINIMUM} <#{MLX_TESTED_BELOW}, but Homebrew has mlx #{mlx_version}.

        MLX changes numerical and autodiff behavior in patch releases, so an
        untested version can break training in ways that only show up mid-run.

        Either install a supported mlx, or if #{mlx_version} is known good, widen
        MLX_TESTED_BELOW in packaging/homebrew/mixlab.rb in the mixlab repository
        after running:
          CGO_ENABLED=1 go test -tags mlx ./arch/... ./gpu ./train -count=1
      EOS
    end

    mlx_prefix = formula_opt_prefix("mlx")

    ENV["CGO_ENABLED"] = "1"
    ENV.append "CGO_CFLAGS", "-I#{mlx_prefix}/include"
    ENV.append "CGO_CXXFLAGS", "-I#{mlx_prefix}/include -std=c++20"
    ENV.append "CGO_LDFLAGS", "-L#{mlx_prefix}/lib -Wl,-rpath,#{mlx_prefix}/lib"

    system "go", "build", "-tags", "mlx",
           "-o", bin/"mixlab", "./cmd/mixlab"
  end

  # Runs without a GPU: the version the linker stamped, then a config parsed and
  # lowered to IR, which also proves the MLX library loads.
  test do
    assert_match "mixlab v#{version} ", shell_output("#{bin}/mixlab -version") unless version.head?
    (testpath/"model.json").write <<~JSON
      {"model_dim": 16, "vocab_size": 32, "seq_len": 4,
       "blocks": [{"type": "plain", "heads": 2}], "training": {"batch_tokens": 4}}
    JSON
    assert_match "valid config", shell_output("#{bin}/mixlab -mode validate -config #{testpath}/model.json")
  end
end
