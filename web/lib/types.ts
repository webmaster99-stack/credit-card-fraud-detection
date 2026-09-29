// Mirrors api/schemas.py. Keep the two in sync by hand; there is no shared schema generator.

export const CATEGORIES = [
  "entertainment",
  "food_dining",
  "gas_transport",
  "grocery_net",
  "grocery_pos",
  "health_fitness",
  "home",
  "kids_pets",
  "misc_net",
  "misc_pos",
  "personal_care",
  "shopping_net",
  "shopping_pos",
  "travel",
] as const;

export interface Transaction {
  trans_ts: string; // ISO datetime, e.g. "2020-11-05T18:42:10"
  amt: number;
  category: string;
  gender: "F" | "M";
  state: string;
  city_pop: number;
  dob: string; // ISO date, e.g. "1985-03-02"
  lat: number;
  long: number;
  merch_lat: number;
  merch_long: number;
  card_id: string;
}

export interface Reason {
  feature: string;
  label: string;
  value: string;
  contribution: number;
  direction: "raises" | "lowers";
  text: string;
}

export interface PredictResult {
  request_id: string;
  fraud_probability: number;
  flagged: boolean;
  threshold: number;
  reasons: Reason[];
  model_name: string;
  model_version: string;
  pipeline_version: string;
}

export interface BatchRowResult {
  row: number;
  request_id: string;
  fraud_probability: number;
  flagged: boolean;
  top_reason: string | null;
}

export interface BatchResult {
  n_rows: number;
  n_flagged: number;
  threshold: number;
  model_name: string;
  model_version: string;
  pipeline_version: string;
  results: BatchRowResult[];
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  alias: string;
  pipeline_version: string;
  feature_set: string;
  step: string;
  calibration: string;
  threshold: number;
  min_precision: number;
  validation: { precision: number; recall: number };
  dataset_name: string;
  dataset_version: string;
  split_spec: string;
  git_commit: string;
}

export interface HealthInfo {
  status: string;
  model_loaded: boolean;
  database_ok: boolean;
}

/** The API's InputValidationError.problems, or a list of Pydantic error messages. */
export interface ApiProblem {
  detail: string[] | string;
}
