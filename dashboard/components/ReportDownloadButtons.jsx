"use client";

import { useState } from "react";

/**
 * 리포트 다운로드 버튼 (PDF + Markdown).
 *
 * - PDF: 비즈니스 보고서 (운영자용)
 * - MD : 기술 리포트 (개발자용)
 *
 * 백엔드가 죽었거나 리포트 미생성이면 사용자에게 안내 표시.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function ReportDownloadButtons() {
  const [status, setStatus] = useState({ pdf: "idle", md: "idle" });
  const [errorMessage, setErrorMessage] = useState("");

  const download = async (kind) => {
    setStatus((prev) => ({ ...prev, [kind]: "loading" }));
    setErrorMessage("");

    try {
      const endpoint = kind === "pdf" ? "/api/v1/report/download/pdf" : "/api/v1/report/download/md";
      const response = await fetch(`${API_BASE}${endpoint}`);

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("리포트가 아직 생성되지 않았습니다. 먼저 분석을 실행해주세요.");
        }
        throw new Error(`서버 오류 (${response.status})`);
      }

      const blob = await response.blob();
      const filename = kind === "pdf" ? "KTX_공실분석_리포트.pdf" : "technical_report.pdf";

      // 브라우저 다운로드 트리거
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setStatus((prev) => ({ ...prev, [kind]: "success" }));
      // 2초 후 idle 복귀
      setTimeout(() => setStatus((prev) => ({ ...prev, [kind]: "idle" })), 2000);
    } catch (error) {
      setStatus((prev) => ({ ...prev, [kind]: "error" }));
      setErrorMessage(error.message);
      setTimeout(() => {
        setStatus((prev) => ({ ...prev, [kind]: "idle" }));
        setErrorMessage("");
      }, 4000);
    }
  };

  const buttonClass = (state) =>
    [
      "flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold transition",
      state === "loading"
        ? "border-slate-300 bg-slate-100 text-slate-400 cursor-wait"
        : state === "success"
        ? "border-emerald-300 bg-emerald-50 text-emerald-700"
        : state === "error"
        ? "border-red-300 bg-red-50 text-red-700"
        : "border-slate-200 bg-white text-slate-600 hover:border-[#72ddb0] hover:bg-[#e5fbf0] hover:text-slate-900",
    ].join(" ");

  const renderLabel = (kind, state) => {
    if (state === "loading") return "생성 중...";
    if (state === "success") return "다운로드 완료";
    if (state === "error") return "오류";
    return kind === "pdf" ? "비즈니스 리포트" : "기술 리포트";
  };

  const renderIcon = (state) => {
    if (state === "loading") return "⏳";
    if (state === "success") return "✓";
    if (state === "error") return "⚠";
    return "↓";
  };

  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => download("pdf")}
          disabled={status.pdf === "loading"}
          className={buttonClass(status.pdf)}
          title="비즈니스 리포트 (운영자용 PDF)"
        >
          <span>{renderIcon(status.pdf)}</span>
          <span>{renderLabel("pdf", status.pdf)}</span>
        </button>
        <button
          type="button"
          onClick={() => download("md")}
          disabled={status.md === "loading"}
          className={buttonClass(status.md)}
          title="기술 리포트 (개발자용 PDF)"
        >
          <span>{renderIcon(status.md)}</span>
          <span>{renderLabel("md", status.md)}</span>
        </button>
      </div>
      {errorMessage && (
        <p className="text-[10px] text-red-600 text-right max-w-[260px]">
          {errorMessage}
        </p>
      )}
    </div>
  );
}