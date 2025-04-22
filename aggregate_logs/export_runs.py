import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


log_dir = "../logs" 

# The RGB runs (/home/lord225/pyrepos/explain-rl/preserve/20250402-092247-StopOpenEarly_6_v3.1 is result or AudianceYorYes)
selected_runs = [
    '20250401-154655-AudienceYourYesv3.1_0',
    '20250330-141743-HisThousandRadiov3.1_0',
    '20250329-212843-BodyAppearInterestingv3.1_0',
    '20250328-173002-ItsUnderEmployeev3.1_0',
    '20250328-130705-StorePersonalOpenv3.1_0',
    '20250327-215233-HappenSoonBookv3.1_0',
    '20250327-161226-LetterBreakAgainstv3.1_0',
    '20250326-140543-DrugRedMessagev3.1_0',
    '20250324-135715-MoreAcceptAgencyv3.1_0',
    '20250323-171342-SaveDegreeBillionv3.1_0',
]

# no regularization runs 20250319-212023-ParentKnowShakev3.1_0
selected_runs = [
    '20250319-212023-ParentKnowShakev3.1_0',
    '20250319-171116-FireActivityTypev3.1_0',
    '20250319-171103-PeacePressureWallv3.1_0',
    '20250319-171103-PeacePressureWallv3.1',
    '20250319-115423-DogEatFillv3.1_0',
    '20250318-150044-ApproachRepresentResourcev3.1',
    '20250318-145643-SpringBagStudentv3.1_0',
    '20250317-171036-SongHardLikev3.1_0',
    '20250317-110955-CareerOptionProjectv3.1_0',
]

selected_runs = [
    '20250408-094532-FlyMarriageResearchvCNN_v2.0_0',
]

# linear mapping runs
selected_runs = [
    "20250418-212536-WhenFactBev4.2_0",
    "20250417-080552-LineShakeLowv4.2_0",
    "20250416-123612-SignWithoutDecadev4.2_0",
    "20250415-154443-OldArticleGardenv4.2_0",
]

output_csv = "linear_results.csv"

def load_scalars_from_run(run_path, run_name):
    ea = EventAccumulator(run_path)
    ea.Reload()
    scalar_data = []

    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        for event in events:
            scalar_data.append({
                "run": run_name,
                "step": event.step,
                "tag": tag,
                "value": event.value,
                "wall_time": event.wall_time,
            })

    return scalar_data

all_data = []
for run_name in selected_runs:
    run_path = os.path.join(log_dir, run_name)
    if os.path.exists(run_path):
        print(f"Processing: {run_name}")
        all_data.extend(load_scalars_from_run(run_path, run_name))
    else:
        print(f"Skipping missing run: {run_name}")

df = pd.DataFrame(all_data)

pivot_df = df.pivot_table(index=["run", "step"], columns="tag", values="value").reset_index()

pivot_df.to_csv(output_csv, index=False)
