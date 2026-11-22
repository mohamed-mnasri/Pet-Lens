# Dataset design

How PetLens gets its training and test data, and why. [PLAN.md](PLAN.md) has the order of work; this file explains the choices behind Steps 2 to 4 and 8.

**Last checked:** 2026-11-22. Counts and licences below come from the linked pages on that date.

## The short version

- **Oxford-IIIT Pet is the core.** It's small but it has the cleanest labels and a clear licence.
- **Wikimedia Commons is the expansion source.** Every file has its own licence and author, and the API is the sanctioned way to fetch it.
- **I review the hard cases myself, not every image.** The model pre-screens Commons images and my own photos and videos. I only check what it disagrees with, plus a random sample.
- **The app says "looks like", never "is".** It answers "domestic cat, no clear breed" when the evidence isn't there, because most pet cats have no breed.

## Oxford-IIIT Pet: what it gives me

| | Cats | Dogs |
| --- | --- | --- |
| Breeds | 12 | 25 |
| Images | 2,371 | 4,978 |
| Per breed | 184 to 200 | 189 to 200 |

Total 7,349. The official split is 3,680 train+val and 3,669 test. These counts are from Table 1 of Parkhi et al. 2012 (linked from the [dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/)) and the [timm dataset card](https://huggingface.co/datasets/timm/oxford-iiit-pet). The licence is CC BY-SA 4.0 and the image copyright stays with the original owners ([Academic Torrents listing](https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1)).

Cat breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx. The dog side includes American Pit Bull Terrier and Staffordshire Bull Terrier.

**Good:** the labels were checked by people, cross-breeds were removed and the classes are balanced. It also has head boxes and trimaps, so I can make body crops that look like what the detector will hand the classifier.

**Not good enough on its own:**

- There's no domestic shorthair or longhair class. Every cat gets forced into a pedigree breed. That's the biggest real-world gap, bigger than the breed count.
- The photos are posed, centred and well lit, and they come from breed enthusiasts. Webcam frames look nothing like them.
- There's no cat identity or photographer metadata, so I can't split by either.
- [Visual Layer's audit](https://huggingface.co/datasets/visual-layer/vl-oxford-iiit-pets) found 75 duplicates and 23 train/test leaks. I dedupe it myself.

## Other datasets I looked at

| Dataset | What it has | Why it's in or out |
| --- | --- | --- |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) | 20,580 images, 120 dog breeds | Built from ImageNet, whose [terms](https://image-net.org/accessagreement) allow non-commercial research and education only. Optional, dogs only. |
| [Tsinghua Dogs](https://cg.cs.tsinghua.edu.cn/ThuDogs/) | 70,428 images, 130 dog breeds | No licence stated, only a request to cite. Out for now. |
| ImageNet cat classes | tabby, tiger cat, Persian, Siamese, Egyptian cat | Two of those are coat patterns, not breeds, and the terms are non-commercial. Out. |
| [Kaggle "Cat Breeds" (ma7555)](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset) | 67 breeds from adoption listings | The labels were written by people listing cats for adoption, and the source site's [terms](https://www.petfinder.com/terms-of-service/) ban automated retrieval. Out. |
| [FeliBreed](https://data.mendeley.com/datasets/ng6tv57j5c/2) | 300 images: Persian, Bengal, "Tabby", Mixed, Local | CC BY 4.0. Too small and unbalanced to train on. Maybe useful as extra test images. |
| [TheCatAPI](https://thecatapi.com/data-licensing-conditions) | Breed images | Needs a paid subscription to train on. Out (€0 rule). |
| Roboflow and Hugging Face re-uploads | Various | The origin and licence usually can't be traced. Out. |

I found no open cat-breed dataset that beats Oxford on both label quality and a licence I can verify.

## Wikimedia Commons

Files under each breed category, including subcategories (`deepcat:` search, bitmaps only), checked through the Commons API on 2026-11-22:

| Category | Files |
| --- | --- |
| Domestic short-haired cats | 9,245 |
| Persian cats | 1,640 |
| British Shorthair cats | 1,067 |
| Norwegian Forest Cats | 884 |
| Sphynx (cat) | 849 |
| Siamese cats | 781 |
| Maine Coon cats | 640 |
| Bengal cats | 482 |
| Abyssinian cats | 423 |
| Ragdoll cats | 361 |
| Scottish Fold cats | 336 |
| American Pit Bull Terrier | 301 |
| Birman cats | 197 |
| Russian Blue cats | 175 |
| Devon Rex | 159 |
| Staffordshire Bull Terrier | 143 |
| Exotic Shorthair cats | 94 |
| Egyptian Mau | 82 |
| Bombay cats | 62 |

Those numbers are upper bounds. Subcategories include drawings, kittens, close-ups of heads and many photos of the same cat, so the usable count after filtering will be lower.

In a sample of 50 Ragdoll files, the licences were CC BY-SA 4.0 (16), CC BY 2.0 (11), CC BY-SA 3.0 (7), CC BY-SA 2.0 (5), public domain (5), CC BY 4.0 (4) and CC0 (2). The 50 files came from 29 different uploaders. All of those licences allow reuse with attribution.

**Rules for using it:**

- Use the [MediaWiki API](https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy) with a User-Agent that names the project and gives contact details. Throttle requests and cache everything, so a rerun doesn't fetch the same file twice.
- Allowlist CC0, public domain, CC BY and CC BY-SA. Anything else gets skipped.
- Store the author, licence and file URL in the manifest, and generate an attribution list from it. Images never go into the repo.
- Treat category names as weak labels. Commons categories contain mislabels and owners' guesses.
- Group by uploader when splitting. One uploader's 40 photos are often one cat.

## Breed coverage and the decision policy

1. The detector finds cats and dogs. Species comes from the detector, and that's the reliable part.
2. A per-species head scores the crop: 12 cat breeds and 25 dog breeds to start with. A cat can never be called a beagle.
3. A distance check in embedding space catches breeds the model hasn't seen.
4. Calibrated confidence picks one of three outcomes. **Accept** shows "looks like a Ragdoll, 87%" with the top 3. **Abstain** says "domestic cat, no clear breed". **Review** queues the prediction for me.

For close pairs (Bengal and Egyptian Mau, Birman and Ragdoll, British Shorthair and Russian Blue, American Pit Bull Terrier and Staffordshire Bull Terrier), the UI shows both names when the scores are close. Bengal vs Egyptian Mau is a confusion documented in the Oxford paper. The others are pairs I expect to be hard but haven't measured yet.

**When a breed gets added:** at least 150 checked training images from 3 or more sources, plus 50 test images from a source the model never trained on. On the Commons numbers above, domestic shorthair, Norwegian Forest Cat and Scottish Fold look like the first candidates. Devon Rex and Exotic Shorthair probably don't have enough images after filtering.

## How much data

These are estimates for a linear head or small MLP on frozen DINOv2 features. They don't guarantee any accuracy.

| Tier | Cat classes | Training images per class | Test images per class |
| --- | --- | --- | --- |
| Prototype | 12 | 70 to 100 (Oxford) | 100 Oxford test |
| Portfolio | 14 (adds domestic shorthair and longhair) | about 250, from 3+ sources | 100 in-domain + 50 unseen-source |
| Ambitious | 20 to 25 | 500+, from 5+ sources | 100 in-domain + 100 unseen-source |

The test set size matters more than it looks. A per-class recall of 0.8 measured on 30 images is only good to about ±14 points. On 50 images it's about ±11, and on 100 about ±8 (95% interval). Below 50 test images, per-breed numbers are mostly noise.

Augmented copies don't count toward any of these numbers. Flipping a photo of one Ragdoll on one sofa doesn't add a new cat, a new room or a new camera.

I'll run a learning curve before collecting more. If in-domain accuracy has stopped improving but unseen-source accuracy is still low, the fix is more variety (new sources, cats and conditions), not more images of the same kind.

## Splits

All images that share a group id go into the same split. The group id is, in order of preference: the cat, then the video, then the near-duplicate cluster, then the Commons uploader. Splits are stratified by class, use a fixed seed, and never reassign an image once it has a split.

| Split | Source | Used for |
| --- | --- | --- |
| Train | Oxford train+val, plus screened Commons images | Fitting the head |
| Validation | Same sources as train, different groups | Choosing the model |
| Calibration | Same sources as train, different groups | Temperature scaling and thresholds |
| In-domain test | Oxford official test, deduplicated against train | A benchmark comparable to published results |
| Unseen-source test | Held-out Commons uploaders, my own footage | Measuring generalisation |
| Golden set | Small, frozen, checked by hand | The promotion gate; never trained on |

I report every metric on each test set separately: macro-F1, per-class recall with intervals, top-3, ECE, and the risk-coverage curve. Pooled test numbers would hide exactly the gap I care about.

## Preprocessing and augmentation

- Train on detector-style crops with a 10% margin, padded to square, at 224 px. Fix EXIF orientation once, during preparation.
- **Use:** random resized crop (scale 0.6 to 1.0), horizontal flip, rotation up to ±15°, mild brightness and contrast jitter, downscaling and JPEG compression to imitate a 720p webcam, random erasing for occlusion.
- **Avoid:** hue shifts and greyscale, because coat colour identifies Russian Blue, Siamese points and Bombay. Also avoid vertical flips, and crops that cut off the ears or face.

## Human-in-the-loop

I don't hand-label datasets. Instead, the model does the first pass and I check what matters:

- **Commons screening.** If a Commons image's category is in the model's top 3, it joins the training pool. If it isn't, it goes to my review queue. A random 5% of the agreements also go to review, which tells me how noisy the agreed labels are. This filter favours easy images, which is why disagreements get reviewed instead of dropped.
- **Batch validation.** I drop photos and videos into an inbox folder and the pipeline predicts on them. On the review page I mark each prediction *right*, *wrong* or *not sure*. On *wrong* I pick the correct answer from the top 3, "domestic / no clear breed", "not a cat or dog", or I type it.
- **Rules that keep the loop honest:**
  - Only reviewed labels enter a new dataset version.
  - The model's own unreviewed predictions never become training labels.
  - Audit and golden-set images are never trained on.
  - My own cats are capped at 20 images per cat per dataset version, and grouped by identity.
  - Every label records where it came from: Oxford, a Commons category, me, or the auditor.
  - A retrained model is only promoted if it beats the current one on the golden set and the unseen-source test.

## Input limits and storage

These limits are enforced at the API, and the full table is in PLAN.md Step 6. Originals are never kept. Every photo and sampled frame is resized and re-encoded on ingest.

- **Photos:** uploads up to 15 MB and 40 megapixels. Each one is stored once at 1024 px on the long edge, as WebP at quality 80, capped at 250 KB. There's also a 384 px crop per animal, capped at 40 KB.
- **Videos:** uploads up to 50 MB and 30 seconds. One frame per second is kept, each stored like a photo. There's no frame-selection logic: it's up to me to upload a short, clean clip showing the whole cat. The clip's answer is the average over its frames.
- **Live camera:** never recorded. At most one frame every 10 seconds is kept while an animal is in view.
- **Review batches:** up to 200 files, reviewed from the compressed copies.

At the caps, 100 photos take under 30 MB and a 30-second clip under 9 MB. Typical files should be smaller, probably 100 to 150 KB per photo (an estimate I'll check in Step 6). Without compression, 100 phone photos would be about 1 GB.

## Pipeline

`make data` runs these stages in order. Each stage can be rerun safely and only does new work:

1. fetch
2. extract
3. ingest: decode, hash, then store as 1024 px WebP (same rules as uploads)
4. normalise labels against `breeds.yaml`
5. detect and crop
6. quality checks
7. embed
8. deduplicate
9. label audit using out-of-fold predictions
10. split
11. report

Layout:

```
D:\petlens-data\
  raw\<source>\<version>\        untouched downloads
  processed\images\<sha[:2]>\     normalised images, named by SHA-256
  processed\embeddings\           <model>\<dataset_version>.parquet
  manifests\<dataset_version>\    manifest.parquet, report.md, checksums.sha256
  inbox\<batch>\                  my photos and videos waiting for a review run
repo\data\
  sources.yaml   breeds.yaml   manifests\<dataset_version>.parquet
```

Manifest columns: `image_id`, `source`, `source_ref`, `licence`, `attribution`, `species`, `breed`, `label_origin`, `group_id`, `cat_id`, `video_id`, `frame_idx`, `width`, `height`, `phash`, `bbox`, `flags`, `split`, `added_in`.

`dataset_version` is the date plus a short hash of the manifest. Every model in the registry records the dataset version it was trained on.

## Still open

- Whether to add Stanford Dogs under its non-commercial terms.
- The DINOv2 licence for weights. I believe it's Apache 2.0, but I'll confirm on the repo before Step 3.
- The real usable Commons counts per breed after filtering. Step 2's report will show them.
