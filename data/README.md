# Data

| Folder | In git? | What goes there |
|---|---|---|
| `public/` | no (ignored) | public datasets, downloaded by each of us. The repo is public, so we don't re-upload them. |
| `collected/` | yes | OBD-II logs from our own cars (issue #35) |

Code finds data by path relative to the repo (`obdfault.config.DATA`), so keep these folder names.

## Public datasets

- **Kaggle "OBD-II datasets"** by cephasax: <https://www.kaggle.com/datasets/cephasax/obdii-ds3> (needs a Kaggle login).
  Put the three `exp*.csv` files in `public/kaggle-obd2/`. Every `obdfault` command reads it from there by default.

## Before committing logs to `collected/`

The repo is public, and so is everything committed here:
- remove GPS / location columns, VINs, and anything else that identifies a person or a car;
- keep each file under 100 MB (GitHub rejects bigger files); split long logs by drive.
