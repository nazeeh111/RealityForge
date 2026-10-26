# License boundaries

RealityForge's original source, generated sample videos, synthetic training data generator and newly trained estimator weights are MIT licensed. Dependencies retain their licenses. No upstream research implementation or pretrained weights are vendored.

**Not every file in this distribution is MIT:**

- `samples/iris/` and `samples/blind-iris/`: IRIS experimental recordings and parameter metadata, CC BY-NC 4.0, credited to Rasul Khanbayov, Mohamed Rayan Barhdadi, Erchin Serpedin and Hasan Kurban. Research/noncommercial use only under that license. See the folder NOTICE and `samples/provenance.json`.
- `samples/independent/`: Rhetos / Mathe-AC Lernwerkstatt Aachen recording, CC BY-SA 4.0. Its trim/transcode remains CC BY-SA 4.0. See the folder NOTICE.
- Screenshots or videos that include either recording must honor that recording's attribution and license. The demo video uses only the original MIT generated fixture, clearly labeled synthetic.

For a code-only MIT distribution, exclude third-party media/metadata and any images or video containing their footage, and remove their sample-manifest entries. The application can use personal uploads and generated fixtures without those assets. Never imply that the IRIS dataset's noncommercial restriction applies to the original MIT source, or that the source's MIT license relicenses the experimental recordings.

Research references: PhysTwin (MIT), SimFoundry (Apache-2.0), IRIS benchmark code (MIT). Their methods informed design choices; no copyright notice for copied source is required because none is incorporated. See `docs/research.md` for inspected revisions and concrete use.
