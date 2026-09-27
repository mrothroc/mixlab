package enrollment

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
)

func TestProtectedProvisioningFileLifecycle(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	p := path(t, statehome.Enrollment)
	check(t, WriteInvitation(p, "node.json", i, now))
	info, err := os.Stat(filepath.Join(p.Dir(), "node.json"))
	check(t, err)
	if info.Mode().Perm() != 0600 {
		t.Fatal("provisioning file is not private")
	}
	r, err := ReadInvitation(p, "node.json", now)
	check(t, err)
	defer r.Clear()
	if !bytes.Equal(r.Secret, i.Secret) || r.ID != i.ID {
		t.Fatal("provisioning round trip changed")
	}
	if err := WriteInvitation(p, "node.json", i, now); err == nil {
		t.Fatal("provisioning file overwritten")
	}
	other := f.invite(t, NodeEnrollment)
	defer other.Clear()
	if err := DestroyInvitation(p, "node.json", other); err == nil {
		t.Fatal("deleted unrelated provisioning file")
	}
	check(t, DestroyInvitation(p, "node.json", i))
	if _, err := os.Stat(filepath.Join(p.Dir(), "node.json")); !os.IsNotExist(err) {
		t.Fatal("consumed secret retained", err)
	}
}

func TestProvisioningFileRejectsPermissionsAndLinks(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	p := path(t, statehome.Enrollment)
	check(t, WriteInvitation(p, "node.json", i, now))
	check(t, os.Chmod(filepath.Join(p.Dir(), "node.json"), 0644))
	if r, err := ReadInvitation(p, "node.json", now); err == nil {
		r.Clear()
		t.Fatal("public secret accepted")
	}
	check(t, os.Chmod(filepath.Join(p.Dir(), "node.json"), 0600))
	check(t, os.Symlink("node.json", filepath.Join(p.Dir(), "alias.json")))
	if r, err := ReadInvitation(p, "alias.json", now); err == nil {
		r.Clear()
		t.Fatal("secret symlink followed")
	}
}
