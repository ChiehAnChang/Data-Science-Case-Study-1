## Dictionary: Traffic Data (Government of Canada) — Data Dictionary (Condensed)

| Feature name | Definition |
|-------------|------------|
| ide_sectn_trafc | Unique identifier for the traffic section. |
| num_sectn_trafc | Traffic section number (the same number is used for both sides of divided roads). |
| des_debut_sous_route | Description of the start of the sub-route. |
| des_fin_sous_route | Description of the end of the sub-route. |
| rtss_debut_chaing | Start RTSS and chainage (in metres) for the traffic section. |
| rtss_fin_chaing | End RTSS and chainage (in metres) for the traffic section. |
| annee_en_cours | Aggregated values for the current year, including DJMA (annual average daily traffic), DJMH (winter AADT), DJME (summer AADT), percentage of heavy vehicles, and the 30th highest hour. |
| anneex | Aggregated values for year "x", including DJMA (annual average daily traffic), DJMH (winter AADT), DJME (summer AADT), and the percentage of heavy vehicles. |
| dat_debut_sectn_trafc | Start date of the traffic section (format: YYYYMMDD). |
| rtss_debut | Start RTSS for the traffic section. RTSS is a 14-character alphanumeric identifier in the MTQ linear reference system, structured as Route [99999], Tronçon [99], Section [999], Sous-route [9x9x]. |
| val_chang_debut | Start chainage of the traffic section (in metres). |
| rtss_fin | End RTSS for the traffic section. RTSS is a 14-character alphanumeric identifier in the MTQ linear reference system, structured as Route [99999], Tronçon [99], Section [999], Sous-route [9x9x]. |
| val_chang_fin | End chainage of the traffic section (in metres). |
| djma_annee_x | Reference year for DJMA (annual average daily traffic) for year "x". |
| val_djma_annee_x | DJMA (annual average daily traffic) value for year "x" (unit: vehicles/day). |
| djme_annee_x | Reference year for DJME (summer average daily traffic) for year "x". |
| val_djme_annee_x | DJME (summer average daily traffic) value for year "x" (unit: vehicles/day). |
| djmh_annee_x | Reference year for DJMH (winter average daily traffic) for year "x". |
| val_djmh_annee_x | DJMH (winter average daily traffic) value for year "x" (unit: vehicles/day). |
| cam_annee_x | Reference year for percentage of heavy vehicles for year "x". |
| val_cam_annee_x | percentage of heavy vehicles value for year "x" (unit: %). |
| val_30e_heure | Design hour (30th highest hour): estimate of the maximum 'normal' hourly traffic flow for the year. |
| index_agreg | Flag indicating whether an aggregated historical data file is available for this section (O=Yes, N=No). |
| index_sectn | Flag indicating whether annual report files for permanent counting sites are available (O=Yes, N=No). |
| index_donnees | Flag indicating whether hourly data files (average hourly by day of week) are available (O=Yes, N=No). |
| url_index_agregees | URL linking to the aggregated historical data file (PDF), when available. |
| url_index_section | URL linking to annual report data files (PDF/XLS), when available. |
| url_index_donnees | URL linking to hourly data files (XLS), when available. |
| objectid | Unique internal identifier. |
