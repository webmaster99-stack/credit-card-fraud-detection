"use client";

import { FormEvent, useState } from "react";

import { VerdictCard } from "@/components/VerdictCard";
import { problemsFrom } from "@/lib/messages";
import { CATEGORIES, type PredictResult, type Transaction } from "@/lib/types";
import { useSlowLoading } from "@/lib/useSlowLoading";

const INITIAL: Transaction = {
  trans_ts: "2020-11-05T18:42:10",
  amt: 54.2,
  category: "grocery_pos",
  gender: "F",
  state: "IL",
  city_pop: 116250,
  dob: "1985-03-02",
  lat: 39.8,
  long: -89.64,
  merch_lat: 39.85,
  merch_long: -89.7,
  card_id: "demo-card-1",
};

const field = "block text-sm";
const input =
  "mt-1 w-full rounded border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-900";

export default function SingleTransactionPage() {
  const [txn, setTxn] = useState<Transaction>(INITIAL);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const slow = useSlowLoading(loading);

  function update<K extends keyof Transaction>(key: K, value: Transaction[K]) {
    setTxn((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setProblems(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(txn),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok || body === null) {
        setProblems(problemsFrom(body, res.status));
      } else {
        setResult(body as PredictResult);
      }
    } catch {
      setProblems(["Could not reach the API. Is it running and is FRAUD_API_URL set?"]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <form onSubmit={onSubmit} className="space-y-3">
        <h1 className="text-xl font-semibold">Score a transaction</h1>

        <label className={field}>
          Card ID
          <input
            className={input}
            value={txn.card_id}
            onChange={(e) => update("card_id", e.target.value)}
            required
          />
        </label>

        <label className={field}>
          Amount (USD)
          <input
            type="number"
            step="0.01"
            min="0.01"
            className={input}
            value={txn.amt}
            onChange={(e) => update("amt", Number(e.target.value))}
            required
          />
        </label>

        <label className={field}>
          Merchant category
          <select
            className={input}
            value={txn.category}
            onChange={(e) => update("category", e.target.value)}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label className={field}>
          Transaction date and time
          <input
            type="datetime-local"
            step="1"
            className={input}
            value={txn.trans_ts}
            onChange={(e) => update("trans_ts", e.target.value)}
            required
          />
        </label>

        <label className={field}>
          Customer date of birth
          <input
            type="date"
            className={input}
            value={txn.dob}
            onChange={(e) => update("dob", e.target.value)}
            required
          />
        </label>

        <label className={field}>
          Customer gender
          <select
            className={input}
            value={txn.gender}
            onChange={(e) => update("gender", e.target.value as Transaction["gender"])}
          >
            <option value="F">F</option>
            <option value="M">M</option>
          </select>
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label className={field}>
            Customer state (2 letters)
            <input
              className={input}
              value={txn.state}
              maxLength={2}
              onChange={(e) => update("state", e.target.value.toUpperCase())}
              required
            />
          </label>
          <label className={field}>
            Customer city population
            <input
              type="number"
              min="0"
              className={input}
              value={txn.city_pop}
              onChange={(e) => update("city_pop", Number(e.target.value))}
              required
            />
          </label>
          <label className={field}>
            Customer latitude
            <input
              type="number"
              step="any"
              className={input}
              value={txn.lat}
              onChange={(e) => update("lat", Number(e.target.value))}
              required
            />
          </label>
          <label className={field}>
            Customer longitude
            <input
              type="number"
              step="any"
              className={input}
              value={txn.long}
              onChange={(e) => update("long", Number(e.target.value))}
              required
            />
          </label>
          <label className={field}>
            Merchant latitude
            <input
              type="number"
              step="any"
              className={input}
              value={txn.merch_lat}
              onChange={(e) => update("merch_lat", Number(e.target.value))}
              required
            />
          </label>
          <label className={field}>
            Merchant longitude
            <input
              type="number"
              step="any"
              className={input}
              value={txn.merch_long}
              onChange={(e) => update("merch_long", Number(e.target.value))}
              required
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
        >
          {loading ? "Scoring..." : "Score transaction"}
        </button>
        {slow && (
          <p className="text-sm text-amber-700 dark:text-amber-300">
            This is taking a while - the free-tier model service sleeps when idle and can take
            about a minute to wake. Your request is still running.
          </p>
        )}
      </form>

      <div>
        {problems && (
          <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
            <p className="font-medium">Could not score this transaction</p>
            <ul className="mt-1 list-inside list-disc">
              {problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        )}
        {result && <VerdictCard result={result} />}
        {!problems && !result && (
          <p className="text-sm text-slate-500">
            Scores against this card&apos;s stored history from earlier requests, if any - a brand
            new card scores as a first-ever transaction.
          </p>
        )}
      </div>
    </div>
  );
}
