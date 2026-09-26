# Potential Datasets

## Engine fault / failure datasets

### EngineFaultDB
A purpose-built automotive engine fault classification dataset. 55,999 entries, classified into four categories representing different fault types, collected around a widely represented spark-ignition engine under controlled laboratory conditions simulating normal and specific fault scenarios. This is exactly the kind of clean, labeled, multi-class fault dataset you'd want to establish your core detection methodology: controlled experiment, known ground truth, real engine parameters (RPM, emissions, etc. per the correlation analysis in the paper).

- Paper: https://www.researchgate.net/publication/375511821_EngineFaultDB_A_Novel_Dataset_for_Automotive_Engine_Fault_Classification_and_Baseline_Results

### EngineAD
A newer (2026) real-world benchmark, not lab-simulated. Sourced from 25 commercial vehicle engines with over six months of continuous, high-resolution sensor telemetry, with expert-driven labeling distinguishing normal operation from early indicators of engine faults. This is close to ideal for the project's framing because it is labeled with early/incipient fault indicators rather than just binary failed/not-failed, which directly matches the "what does about to break look like" angle.

- Paper: https://arxiv.org/html/2603.25955v1
- Code/data: https://github.com/Armanfard-Lab/EngineAD

### HIL Fault-Injection Dataset (gasoline/EV)
From a fault-injection validation study. Each recording is one experimental run under healthy or faulty conditions at a fixed 0.01s interval, with the gasoline-engine system recording 15 signal columns, evaluated across ten single-fault classes including engine-speed and throttle faults. Good for controlled, high-resolution fault-labeled time series if you want cleaner signal-level ground truth than a real fleet gives you.

- Paper: https://arxiv.org/pdf/2607.03734

## Real-world OBD-II telemetry

### KIT Automotive OBD-II Dataset (RADAR4KIT)
Real trip telemetry from a Seat León and Opel Corsa, 10 OBD-II signals (coolant temp, MAP, RPM, speed, intake air temp, MAF, throttle position, ambient temp, accelerator position). Real-world PID data in the same format our ELM327/OBD reader will output. Good for learning normal sensor ranges/dynamics.

- Link: https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq

### carOBD (GitHub)
Open-source dataset from a Toyota Etios, 27 PIDs at 1Hz via a Carloop OBD-II interface. Good second real-vehicle source for generalization testing.

- Link: https://github.com/eron93br/carOBD

### OBD-Dataset (GitHub)
- Link: https://github.com/AbouAbdallah-Lounis/OBD-Dataset

### Kaggle "OBD-II datasets"
Crowdsourced fleet data, ~47,500 rows / 33 features, includes TROUBLE_CODES/DTC_NUMBER fields we can use as failure labels. Probably our best bet for labeled classification in the actual OBD-II PID format.

- Link: https://www.kaggle.com/datasets (search "OBD-II datasets")

## Supporting / lookup data

### OBD2 Powertrain Codes (Kaggle)
Catalog of P-codes mapped to failure categories; useful as a lookup table to turn raw DTCs into structured failure classes.

- Link: https://www.kaggle.com/datasets/donnetew/odb2-powertrain-codes

## General predictive maintenance benchmarks

### AI4I 2020 Predictive Maintenance Dataset (UCI)
10,000 samples, 14 features, multi-label failure types (tool wear, heat dissipation, power, overstrain, random). Clean tabular benchmark for imbalanced failure classification.

- Link: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### Vehicle Maintenance Data (Kaggle)
Synthetic, 50,000 vehicles, 19 features (mileage, service history, reported issues, accident history) with a binary Need_Maintenance label. Useful for a higher-level "does this car need service soon" layer, though synthetic and coarser than sensor-level data.

- Link: https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data
