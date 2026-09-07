"""Excel (.xlsx) document loader converting workbooks and sheets to text."""

import os
from typing import Any, Dict, List


def load(file_path: str) -> List[Dict[str, Any]]:
    """Loads an Excel spreadsheet and formats all sheets and rows into text.

    Args:
        file_path: Path to the XLSX file.

    Returns:
        List containing {"text": str, "source": str, "page": None}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If openpyxl or pandas are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas and openpyxl are required for loading Excel spreadsheets. "
            "Install via requirements.txt."
        ) from e

    excel_file = pd.ExcelFile(file_path, engine="openpyxl")
    output_sections: List[str] = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        if df.empty:
            continue
        sheet_rows = [f"[Sheet: {sheet_name}]"]
        for idx, row in df.iterrows():
            row_items = [
                f"{col}: {str(val).strip()}"
                for col, val in row.items()
                if pd.notna(val) and str(val).strip() != ""
            ]
            if row_items:
                sheet_rows.append(f"Row {idx + 1}: " + "; ".join(row_items))

        if len(sheet_rows) > 1:
            output_sections.append("\n".join(sheet_rows))

    full_text = "\n\n".join(output_sections)
    return [{"text": full_text, "source": file_path, "page": None}]
