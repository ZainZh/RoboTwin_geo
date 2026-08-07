# Persistent Hammer services

These checked-in units replace transient `systemd-run --user` jobs. The
system-level units are preferred because their training cgroups live outside
`user@1000.service`; the user units are a no-sudo fallback. Never enable both
copies simultaneously. Per-run `flock` prevents duplicate writes, but a second
service would wait unnecessarily and complicate operations.

Before installation, run the read-only gates:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/zheng/miniforge3/envs/RoboTwin/bin/python \
  policy/DP3/scripts/verify_hammer_local_resume.py
systemd-analyze verify policy/DP3/systemd/system/*.service
systemd-analyze --user verify policy/DP3/systemd/user/*.service
```

## Preferred system installation

This requires root only for installation and lifecycle management:

```bash
sudo install -m 0644 policy/DP3/systemd/system/hammer-local-loss@.service \
  policy/DP3/systemd/system/hammer-remote-sync-watcher.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now \
  hammer-local-loss@full_fixed.service \
  hammer-local-loss@no_ce.service \
  hammer-local-loss@no_supcon.service \
  hammer-local-loss@no_consistency.service \
  hammer-remote-sync-watcher.service
```

The system template uses `User=zheng`, `Group=zheng`, `NUM_WORKERS=2`, one
BLAS thread, `MemoryHigh=7G`, `MemoryMax=9G`, `MemorySwapMax=512M`,
`ManagedOOMPreference=avoid`, and `Restart=on-failure`. It is wanted by
`multi-user.target`, so reboot/login/user-manager replacement does not remove
the jobs.

## No-sudo user fallback

```bash
mkdir -p /home/zheng/.config/systemd/user
install -m 0644 policy/DP3/systemd/user/hammer-local-loss@.service \
  policy/DP3/systemd/user/hammer-remote-sync-watcher.service \
  /home/zheng/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now \
  hammer-local-loss@full_fixed.service \
  hammer-local-loss@no_ce.service \
  hammer-local-loss@no_supcon.service \
  hammer-local-loss@no_consistency.service \
  hammer-remote-sync-watcher.service
```

These files survive recreation of the user manager and are pulled back in by
`default.target`. They are still inside `user@1000.service`, so a later oomd
kill of that whole cgroup interrupts them briefly; full-state `resume.pt` plus
`Restart=on-failure`/target activation resumes them without overwriting a
completed run.

To migrate from fallback to system units, disable and stop all five user units
before enabling the system copies. Do not delete run directories or lock files.

## Runtime audit

```bash
systemctl status 'hammer-local-loss@*.service' hammer-remote-sync-watcher.service
journalctl -u 'hammer-local-loss@*.service' -n 100 --no-pager
nvidia-smi
```

Completion is idempotent: the launcher uses `AUTO_RESUME=1`; run-lock validates
`completion.json` and checkpoint hashes, then exits zero instead of retraining.
Changing `num_workers` from the stored 4 to operational value 2 is allowed only
after the stored identity SHA/args/run/seed validate. All scientific arguments,
including the resolved Utonia checkpoint, remain strict.
