from src.dataset.dataset_validator import find_cross_split_leaks, count_images_per_class
for s in ("Training", "Validation", "Testing"):
    c = count_images_per_class(f"Dataset/{s}")
    print(s, sum(c.values()), "images | min/max per class:", min(c.values()), max(c.values()))
leaks = find_cross_split_leaks("Dataset")
print("Identical images across splits:", len(leaks))