wasm-dev:
	cd crates/bitnet-wasm && bash dev.sh

wasm-stop:
	cd crates/bitnet-wasm && bash dev.sh stop

help:
	@echo "Available targets:"
	@echo "  wasm-dev   - Build WASM, start server, and open browser for bitnet-wasm (from root)"
	@echo "  wasm-stop  - Stop the WASM dev server (from root)"
	@echo "  help       - Show this help message" 