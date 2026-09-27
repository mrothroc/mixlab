package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/mrothroc/mixlab/internal/buildinfo"
)

const developmentNotice = "Experimental managed training; trusted administrator-controlled hosts only."

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "internal-worker-host":
			return runInternalWorkerHost(args[1:], stderr)
		case "init":
			return runInit(args[1:], stdout, stderr)
		case "invite":
			return runInvite(args[1:], stdout, stderr)
		case "enroll":
			return runEnroll(args[1:], stdout, stderr)
		case "enrollment":
			if len(args) > 1 && args[1] == "serve" {
				return runEnrollmentServe(args[2:], stdout, stderr)
			}
		case "authority":
			if len(args) > 1 && args[1] == "serve" {
				return runAuthorityServe(args[2:], stdout, stderr)
			}
		case "revoke":
			return runRevoke(args[1:], stdout, stderr)
		case "agent":
			return runAgent(args[1:], stdout, stderr)
		case "nodes":
			return runNodes(args[1:], stdout, stderr)
		case "submit":
			return runSubmit(args[1:], stdout, stderr)
		}
	}
	flags := flag.NewFlagSet("mixlab-cluster", flag.ContinueOnError)
	flags.SetOutput(stderr)
	version := flags.Bool("version", false, "print build and worker protocol identity, then exit")
	help := flags.Bool("help", false, "show this help")
	usage := func(w io.Writer) {
		_, _ = fmt.Fprintln(w, "Usage: mixlab-cluster -version | -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster init|invite|enroll -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster enrollment serve -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster authority serve -help | revoke -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster agent init -help | agent -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster nodes -help")
		_, _ = fmt.Fprintln(w, "       mixlab-cluster submit -help")
		_, _ = fmt.Fprintln(w, developmentNotice)
		flags.SetOutput(w)
		flags.PrintDefaults()
		flags.SetOutput(stderr)
	}
	flags.Usage = func() { usage(stderr) }
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if flags.NArg() != 0 {
		_, _ = fmt.Fprintf(stderr, "mixlab-cluster: unexpected argument %q\n", flags.Arg(0))
		usage(stderr)
		return 2
	}
	if *help {
		usage(stdout)
		return 0
	}
	if *version {
		_, _ = fmt.Fprintln(stdout, buildinfo.Report("mixlab-cluster"))
		_, _ = fmt.Fprintln(stdout, developmentNotice)
		return 0
	}
	usage(stderr)
	return 2
}
