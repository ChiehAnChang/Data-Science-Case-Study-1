In this folder, our developer follows this structure to organize the dataset:

```text
├── Air Pollutant Data
│   ├── Ontario
│   │   └── Dataset.csv
│   ├── British Columbia
│   │   └── Dataset.csv
│   ├── Alberta
│   │   └── Dataset.csv
│   └── Winnipeg Manitoba
│       └── Dataset.csv
├── Meteorological Data
│   └── Canada
├── Traffic Data
│   ├── Statistics Canada
│   │   └── Dataset.csv
│   ├── Government of Canada
│   │   └── Dataset.csv
└── International
    ├── Beijing
    │   └── Dataset.csv
    ├── Shenzhen
    │   └── Dataset.csv
    └── KnowAir
```

In addition to the folder architecture, our dictionary code book follows a two-column table. 
In **Column 1**, the name of the feature is shown for the dataset. **Column 2** contains its corresponding definition and some explanation.

**Example:**

| Features | Definition |
| :--- | :--- |
| `Lon` | A feature representing the longitude of a location in X place |