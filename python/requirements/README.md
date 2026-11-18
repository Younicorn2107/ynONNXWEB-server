# Python requirement sets

`<set>.txt` lists what a set needs at exact versions; `<set>.lock` is what the
server installs: every package, transitive ones included, with the digests of
its published files (`pip install --require-hashes`). Regenerate the locks
after changing a `.txt`, from a clean environment with pip-tools:

    pip-compile --generate-hashes --allow-unsafe --strip-extras -o hub.lock hub.txt
    pip-compile --generate-hashes --allow-unsafe --strip-extras -o export.lock export.txt
    pip-compile --generate-hashes --allow-unsafe --strip-extras -o yolo.lock yolo.txt

Then run one export per set before releasing.
