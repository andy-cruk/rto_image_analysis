# Reverse the Odds
Scripts to analyse histopathological classifications from the Cancer Research UK Citizen Science game Reverse the Odds. At current stage not very useful unless you have direct access to the data, which is not publicly available for now.

For internal use, the following files are required but not available on github:
* classifications and subjects collections imported into MongoDB

## Steps to analyse the data
* Import classifications and subjects into mongoDB, and set the correct db name in `rto_mongodb_utils.py`
* Run `rto_mongodb_utils` with all functions uncommented to preprocess/clean the data (if errors, probably because stain names are inconsistent; track down and add correction to `rto_mongodb_utils.py`)
* Run `user_aggregation.py` for each stain type, once with bootstrap turned off, once turned on (and `aggregate = 'ignoring_segments'`). This will fill up the 'results' database in mongoDB with collections 'bootstraps' and 'cores')
* Run `patient_level.py`'s `pll.combine_cores_per_patient()` to generate a 'patients' collection with each stain aggregated for each patient. This also creates excel files for easy sharing with research groups

Then you can plot results with `core_level_plots.py` or `patient_level.py`

## Copyright / Licence
Copyright 2016/2017 Cancer Research UK

Source Code License: The GNU Affero General Public License, either version 3 of the License or (at your option) any later version. (See agpl.txt file)

The GNU Affero General Public License is a free, copyleft license for software and other kinds of works, specifically designed to ensure cooperation with the community in the case of network server software.

Documentation is under a Creative Commons Attribution Noncommercial License version 3.

