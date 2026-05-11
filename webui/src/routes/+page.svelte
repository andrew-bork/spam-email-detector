<div class="shell">
	<aside>
		<h1>Inference</h1>
	</aside>
	<main>
		<div class="page-title">Run Inference</div>
		<div class="page-sub">Classify text with all models at once, or pick a specific one.</div>

		<!-- Input card -->
		<div class="card">
			<div class="card-title">Input</div>
			<div class="card-title" style="margin-bottom:8px; font-size:10px;">Try an example</div>
			<div class="example-strip">
				<!-- <span class="ex-pill" onclick="loadExample(this)">Congratulations! You've won a $1,000 gift card. Click here NOW to claim your prize.</span>
        <span class="ex-pill" onclick="loadExample(this)">Hi, are we still on for the 3pm sync tomorrow?</span>
        <span class="ex-pill" onclick="loadExample(this)">URGENT: Your account has been compromised. Verify your details immediately to avoid suspension.</span>
        <span class="ex-pill" onclick="loadExample(this)">Can you review the pull request I submitted earlier? No rush at all.</span>
        <span class="ex-pill" onclick="loadExample(this)">Free entry in 2 a wkly comp to win FA Cup tickets! Txt FA to 87121</span>
        <span class="ex-pill" onclick="loadExample(this)">The quarterly report is attached. Let me know if you have questions.</span>
        <span class="ex-pill" onclick="loadExample(this)">You have been selected for an EXCLUSIVE offer. Act now before it expires!</span> -->
			</div>

			<div style="position: relative;">
				<textarea
					id="infer-text"
					placeholder="Paste or type a message to classify…"
					// oninput="onInferInput()"
					rows="4"
				></textarea>
				<span
					id="infer-charcount"
					style="position:absolute; bottom:10px; right:12px; font-size:10px; font-family:var(--font-mono); color:var(--text3); pointer-events:none;"
					>0</span
				>
			</div>
			<div id="token-hint" class="token-preview" style="margin-top:6px;"></div>

			<div class="btn-row">
				<button class="btn btn-run-all" id="run-all-btn" disabled>
					<svg width="13" height="13" viewBox="0 0 16 16" fill="white"
						><path d="M2 4h12M2 8h12M2 12h12" stroke="white" stroke-width="1.5" fill="none" /></svg
					>
					Run All Classifiers
				</button>
				<button class="btn btn-primary" id="infer-btn" disabled>
					<svg width="12" height="12" viewBox="0 0 16 16" fill="white"
						><polygon points="3,2 13,8 3,14" /></svg
					>
					Run Selected
				</button>
				<button class="btn btn-secondary">Clear</button>
			</div>
		</div>

		<!-- Results area -->
		<div id="infer-result-area"></div>
	</main>
</div>

<style>
	/* Layout */
	.shell {
		height: 100%;
		display: grid;
		grid-template-columns: 280px 1fr;
		/* min-height: 100vh;  */
	}

	button {
		height: 36px;
		padding: 0 16px;
		border-radius: var(--radius);
		border: 0.5px solid var(--border2);
		background: transparent;
		color: var(--text);
		font-size: 13px;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		transition:
			background 0.15s,
			opacity 0.15s;
	}
	button:hover {
		background: var(--bg3);
	}
	button:active {
		transform: scale(0.98);
	}
	button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn-primary {
		background: var(--accent-bg);
		border-color: var(--accent);
		color: var(--accent-text);
		font-weight: 500;
	}
	.btn-primary:hover {
		background: color-mix(in srgb, var(--accent-bg) 80%, var(--accent));
	}
	.btn-danger {
		background: var(--danger-bg);
		border-color: var(--danger);
		color: var(--danger);
	}

	main {
		padding: 16px;
	}

	/* Sidebar */
	aside {
		border-right: 0.5px solid var(--border);
		overflow-y: auto;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 20px;
		background: var(--bg2);
	}
</style>
