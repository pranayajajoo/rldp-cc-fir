# AGENTS.md

## Remote Development

This project is hosted on the remote server `fir`.

Run project commands over SSH from the local machine:

```bash
ssh fir 'cd /home/pranayaj/projects/def-whitem/pranayaj/projects/mr_zsrl && <command>'
```

Before running remote commands, check whether the shared SSH connection is active:

```bash
ssh -O check fir
```

If the shared connection is active, continue using `ssh fir 'cd /home/pranayaj/projects/def-whitem/pranayaj/projects/mr_zsrl && <command>'`.

If it is not active, ask the user to run this once and approve the Duo 2FA prompt:

```bash
ssh fir true
```

Do not attempt to bypass Duo, automate Duo, or store Duo passcodes.

The local SSH host entry should include these settings:

```sshconfig
Host fir
  HostName fir.alliancecan.ca
  User pranayaj
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 4h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

## GitHub Tracking

This project is connected to GitHub at `pranayajajoo/rldp-cc-fir`.

The remote configuration should keep the user's fork as `origin` and the source project as `upstream`:

```bash
git remote add origin git@github.com:pranayajajoo/rldp-cc-fir.git
git remote add upstream https://github.com/hari-sikchi/mr_zsrl.git
```

For every user-requested task that changes files, make a focused commit containing only the intended edits and push it to `origin`. Before staging, inspect `git status --short` and avoid staging unrelated modified or untracked files already present in the worktree.

When the user asks to revert changes, first inspect the relevant commit history and diffs, then use a traceable revert strategy. Prefer `git revert <commit>` for already-pushed commits. For uncommitted edits, revert only the specific files or hunks involved in the requested change, and never discard unrelated user work.

After committing and pushing, report the commit hash and pushed branch.
