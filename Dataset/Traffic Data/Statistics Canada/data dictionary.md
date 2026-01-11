## Dictionary: Traffic Flow (Statistics Canada) — Daily Vehicle Counts by Camera (Wide Format)

| Feature name | Definition |
|-------------|------------|
| WKT | Camera location geometry in WKT format: `POINT (longitude latitude)` (degrees). |
| CSDUID | Census Subdivision identifier associated with the camera location. Treat as an ID (string) even if stored numerically. |
| traffic_camera | Unique identifier for the traffic camera (one row per camera in this file). |
| traffic_source | Data source/jurisdiction providing the camera feed (e.g., a province or municipality). |
| camera_road | Road or corridor name associated with the camera, as provided by the source system. |
| xYYYY_MM_DD | Daily vehicle count for the calendar date `YYYY-MM-DD` for the given camera. Column names follow the pattern `xYYYY_MM_DD` (e.g., `x2022_02_02`). This file includes 1368 daily columns spanning 2022-02-02 to 2025-10-31. Values are non-negative counts (stored as numeric; may appear as decimals due to missing values). Missing/blank values indicate no data available for that camera on that date. |
