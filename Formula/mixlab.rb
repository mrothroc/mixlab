class Mixlab < Formula
  desc "ML architecture exploration tool — JSON configs, Go IR, Metal/CUDA"
  homepage "https://github.com/mrothroc/mixlab"
  url "https://github.com/mrothroc/mixlab.git",
      tag:      "v0.107.2",
      revision: "0cda3610350e91c5cfcd25f970732bce5a1ae0d5"
  license "MIT"
  head "https://github.com/mrothroc/mixlab.git", branch: "main"

  depends_on "go" => :build
  depends_on "mlx"
  depends_on :macos

  # Homebrew has no versioned mlx formula and depends_on takes no version
  # predicate, so the tested range is asserted here instead. MLX 0.32.1 changed
  # gather VJP semantics under a patch bump and silently broke MoE and bf16
  # training, so an untested MLX is treated as a hard error rather than allowed
  # through. Widen this after running the -tags mlx suite against the new MLX.
  MLX_TESTED_MINIMUM = "0.32.0"
  MLX_TESTED_BELOW = "0.33.0"

  def install
    mlx_version = Formula["mlx"].version
    if mlx_version < Version.new(MLX_TESTED_MINIMUM) ||
       mlx_version >= Version.new(MLX_TESTED_BELOW)
      odie <<~EOS
        mixlab #{version} is tested against MLX >=#{MLX_TESTED_MINIMUM} <#{MLX_TESTED_BELOW}, but Homebrew has mlx #{mlx_version}.

        MLX changes numerical and autodiff behavior in patch releases, so an
        untested version can break training in ways that only show up mid-run.

        Either install a supported mlx, or if #{mlx_version} is known good, widen
        MLX_TESTED_BELOW in Formula/mixlab.rb after running:
          CGO_ENABLED=1 go test -tags mlx ./arch/... ./gpu ./train -count=1
      EOS
    end

    mlx_prefix = Formula["mlx"].opt_prefix

    ENV["CGO_ENABLED"] = "1"
    ENV.append "CGO_CFLAGS", "-I#{mlx_prefix}/include"
    ENV.append "CGO_CXXFLAGS", "-I#{mlx_prefix}/include -std=c++20"
    ENV.append "CGO_LDFLAGS", "-L#{mlx_prefix}/lib -Wl,-rpath,#{mlx_prefix}/lib"

    system "go", "build", "-tags", "mlx",
           "-o", bin/"mixlab", "./cmd/mixlab"
  end

  test do
    assert_match "PASS", shell_output("#{bin}/mixlab -mode smoke 2>&1")
  end
end
