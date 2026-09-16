# Sample corpus

The Northstar Field Station handbook is an original, fictional corpus created for
GroundedDesk by Santhosh A with AI assistance. It is distributed under this repo's
MIT license. It contains no private documents, real credentials or operating advice.

Upload **fieldguide.pdf** for the demo. `fieldguide.md` is the readable equivalent;
do not upload both when reproducing the benchmark. TXT/MD have logical page 1.
The evaluator indexes only the PDF in a fresh collection, never user uploads.

Rebuild with `python scripts/build_sample.py`. This also regenerates `eval/golden.json`.
The set has 20 supported questions and five unsupported questions. Questions and
answers were authored together, so this is a development smoke benchmark with
substantial lexical overlap, not an independent test set or evidence of real-world accuracy.
