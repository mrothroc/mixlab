// Package statehome implements location and filesystem primitives for private
// persistent state. It does not interpret state or derive authority from paths.
// Callers supply all configuration, including environment and home values.
//
// Filesystem operations support macOS and Linux and fail closed elsewhere.
// The effective user's account is trusted: private directories exclude other
// users, and advisory directory locks serialize cooperating publishers. These
// primitives are not a sandbox against hostile same-user or root processes.
// External writers must use the same locking protocol. All path components must
// be real directories; callers using OS-provided symlink aliases must explicitly
// resolve those aliases before passing configuration to this package.
// Darwin ACLs are inspected natively, without cgo: granting ACEs (including
// inheritance-only grants) and unknown or malformed ACL data are rejected.
// Deny-only ACLs, such as a home directory's deny-delete entry, are permitted.
// ACL inspection failures fail closed. Linux POSIX ACL access masks are checked
// through mode bits, including rechecking all created files and directories.
//
// ReadFile reads the entire protected file. Domain-specific size limits belong
// to callers; this package neither parses payloads nor assigns limits to them.
package statehome
