/* ═══════════════════════════════════════════════════════
   Admin Panel – Client-side Logic
   Camera config save, test connection, reload config
   ═══════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ─── Helper: Show toast notification ───
    function showToast(containerId, message, type) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Clear previous toast
        container.innerHTML = '';

        const icons = {
            success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            error:   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
            info:    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `${icons[type] || icons.info} <span>${message}</span>`;
        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-8px)';
            setTimeout(() => toast.remove(), 400);
        }, 5000);
    }

    // ─── Save Camera Config ───
    const cameraForm = document.getElementById('camera-form');
    if (cameraForm) {
        cameraForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const entryCam = document.getElementById('entry-cam').value.trim();
            const shelfCam = document.getElementById('shelf-cam').value.trim();
            const saveBtn  = document.getElementById('btn-save-cameras');

            if (!entryCam || !shelfCam) {
                showToast('camera-toast', 'Please fill in both camera fields.', 'error');
                return;
            }

            // Disable button and show loading state
            const originalHTML = saveBtn.innerHTML;
            saveBtn.disabled = true;
            saveBtn.innerHTML = '<span class="btn-loader" style="display:inline-block;width:18px;height:18px;border:2.5px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite"></span> Saving…';

            try {
                const response = await fetch('/admin/api/save-cameras', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        entry_cam: entryCam,
                        shelf_cam: shelfCam
                    })
                });

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    showToast('camera-toast', '✓ Camera settings saved to config.local.json — restart the server to apply changes.', 'success');
                } else {
                    showToast('camera-toast', data.message || 'Failed to save settings.', 'error');
                }
            } catch (err) {
                showToast('camera-toast', 'Network error — could not reach server.', 'error');
            } finally {
                saveBtn.disabled = false;
                saveBtn.innerHTML = originalHTML;
            }
        });
    }

    // ─── Test Camera Connection ───
    const testBtn = document.getElementById('btn-test-cameras');
    if (testBtn) {
        testBtn.addEventListener('click', async function () {
            const entryCam = document.getElementById('entry-cam').value.trim();
            const shelfCam = document.getElementById('shelf-cam').value.trim();

            if (!entryCam || !shelfCam) {
                showToast('camera-toast', 'Please fill in both camera fields before testing.', 'error');
                return;
            }

            const originalHTML = testBtn.innerHTML;
            testBtn.disabled = true;
            testBtn.innerHTML = '<span class="btn-loader" style="display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite"></span> Testing…';

            try {
                const response = await fetch('/admin/api/test-cameras', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        entry_cam: entryCam,
                        shelf_cam: shelfCam
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    const entryOk = data.entry_ok ? '✓' : '✗';
                    const shelfOk = data.shelf_ok ? '✓' : '✗';
                    const allOk = data.entry_ok && data.shelf_ok;
                    const type = allOk ? 'success' : 'error';
                    const msg = `Entry Cam: ${entryOk} ${data.entry_ok ? 'Connected' : 'Failed'}  |  Shelf Cam: ${shelfOk} ${data.shelf_ok ? 'Connected' : 'Failed'}`;
                    showToast('camera-toast', msg, type);
                } else {
                    showToast('camera-toast', data.message || 'Test failed.', 'error');
                }
            } catch (err) {
                showToast('camera-toast', 'Network error — could not reach server.', 'error');
            } finally {
                testBtn.disabled = false;
                testBtn.innerHTML = originalHTML;
            }
        });
    }

    // ─── Reload Config ───
    const reloadBtn = document.getElementById('btn-reload-config');
    if (reloadBtn) {
        reloadBtn.addEventListener('click', async function () {
            try {
                const response = await fetch('/admin/api/current-config');
                const data = await response.json();

                if (response.ok) {
                    document.getElementById('entry-cam').value = data.entry_cam || '';
                    document.getElementById('shelf-cam').value = data.shelf_cam || '';
                    document.getElementById('info-host').textContent = data.host_ip || '';
                    document.getElementById('info-port').textContent = data.host_port || '';
                    showToast('camera-toast', 'Config reloaded from config files.', 'info');
                }
            } catch (err) {
                showToast('camera-toast', 'Failed to reload config.', 'error');
            }
        });
    }

})();
