<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { resolve } from '$app/paths';

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<div class="shell">
	<header>
		<div class="dot" id="dot"></div>
		<h1>Spam Detector</h1>
		<nav>
			<a href={resolve('/inference')}>Inference</a>
		</nav>
		<span class="status-text" id="status-text">idle</span>
	</header>
	<div class="main">
		{@render children()}
	</div>
</div>

<style>
	header {
		/* grid-column: 1/-1;  */
		display: flex;
		align-items: center;
		gap: 24px;
		padding: 0px 20px;

		border-bottom: 0.5px solid var(--border2);
		background: var(--bg);
	}
	header h1 {
		font-size: 24px;
		font-weight: 500;
		letter-spacing: -0.01em;
		/* margin: 0 24px; */
	}

	.status-text {
		margin-left: auto;
	}

	.shell {
		display: grid;
		grid-template-rows: 96px 1fr;
		min-height: 100vh;
		max-width: 100vw;
	}
	.main {
		max-width: 100vw;
	}
</style>
