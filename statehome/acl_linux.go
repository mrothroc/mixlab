package statehome

import "os"

// Linux POSIX access ACLs cannot grant beyond the group-class mode mask.
// checkPrivate/checkAncestor enforce that mask. Default ACLs are not access
// grants on the parent; every newly created child is checked again, including
// temporary files before writing and staged trees before publication.
func checkPathACL(string) error   { return nil }
func checkFileACL(*os.File) error { return nil }
