import os
import glob
import pandas as pd
import pymeshlab


def analyze_brain_subjects(
    folder_path="brains",
    output_csv="brain_metrics.csv",
    exposed_suffix="_exposed",
):
    # Find all .stl files except previously generated exposed ones
    all_stl_files = [
        f
        for f in glob.glob(os.path.join(folder_path, "*.stl"))
        if not f.endswith(f"{exposed_suffix}.stl")
    ]

    if not all_stl_files:
        print(f"No .stl files found in '{folder_path}' folder.")
        return

    # Map base subject names to their respective total and white STL paths
    subjects = {}
    for file_path in all_stl_files:
        filename = os.path.basename(file_path)
        name_no_ext, _ = os.path.splitext(filename)

        if name_no_ext.endswith("_white"):
            base_name = name_no_ext[:-6]  # Strip '_white'
            if base_name not in subjects:
                subjects[base_name] = {}
            subjects[base_name]["white"] = file_path
        else:
            base_name = name_no_ext
            if base_name not in subjects:
                subjects[base_name] = {}
            subjects[base_name]["total"] = file_path

    # Filter out subjects that don't have a corresponding _white mesh
    valid_subjects = {
        name: paths
        for name, paths in subjects.items()
        if "white" in paths and "total" in paths
    }

    if not valid_subjects:
        print("No valid subject pairs (matching total and _white files) found.")
        return

    results = []

    for name in sorted(valid_subjects.keys()):
        paths = valid_subjects[name]
        print(f"Processing subject: {name}")

        try:
            # 1. Process Total Surface (Non-white)
            ms_total = pymeshlab.MeshSet()
            ms_total.load_new_mesh(paths["total"])
            geom_total = ms_total.get_geometric_measures()

            area_total = geom_total.get("surface_area", None)
            vol_total = geom_total.get("mesh_volume", None)

            # 2. Process White Matter Surface
            ms_white = pymeshlab.MeshSet()
            ms_white.load_new_mesh(paths["white"])
            geom_white = ms_white.get_geometric_measures()

            area_white = geom_white.get("surface_area", None)
            vol_white = geom_white.get("mesh_volume", None)

            # 3. Compute Convex Hull on Total Surface (Exposed)
            ms_total.generate_convex_hull()

            # Save exposed mesh STL
            ext = os.path.splitext(paths["total"])[1]
            exposed_filename = f"{name}{exposed_suffix}{ext}"
            exposed_filepath = os.path.join(folder_path, exposed_filename)

            # Compute exposed measures
            geom_exposed = ms_total.get_geometric_measures()
            area_exposed = geom_exposed.get("surface_area", None)
            vol_exposed = geom_exposed.get("mesh_volume", None)

            results.append(
                {
                    "name": name,
                    "Area total": area_total,
                    "Volume total": vol_total,
                    "Area white": area_white,
                    "Volume white": vol_white,
                    "Area exposed": area_exposed,
                    "Volume exposed": vol_exposed,
                }
            )

        except Exception as e:
            print(f"Error processing subject '{name}': {e}")

    # Build DataFrame and export to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully saved metrics for {len(results)} subjects to {output_csv}")

if __name__ == "__main__":
    import os
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)
    analyze_brain_subjects()