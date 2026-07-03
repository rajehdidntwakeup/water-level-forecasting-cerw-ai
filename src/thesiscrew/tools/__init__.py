"""CrewAI tools for the water-level forecasting crew."""

from thesiscrew.tools.pegelonline_tool import (
    ListStationsTool,
    StationDetailTool,
    GetMeasurementsTool,
    GetMeasurementsCSVTool,
    GetForecastTool,
    GetWaterBodiesTool,
)
from thesiscrew.tools.open_meteo_tool import (
    HistoricalWeatherTool,
    ForecastWeatherTool,
    KorneuburgWeatherTool,
    KorneuburgForecastTool,
)
from thesiscrew.tools.ehyd_tool import (
    ListAustrianStationsTool,
    StationMetadataTool,
    StationDataTool,
    CharacteristicValuesTool,
)
from thesiscrew.tools.data_processing_tool import (
    ListDataFilesTool,
    CSVSummaryTool,
    ParquetSummaryTool,
    ResampleTool,
    FillGapsTool,
    LagFeaturesTool,
    RollingFeaturesTool,
    CalendarFeaturesTool,
    RateOfChangeTool,
    ChronoSplitTool,
    ComputeMetricsTool,
)
from thesiscrew.tools.model_evaluation_tool import (
    PersistenceBaselineTool,
    WalkForwardTool,
    StratifiedMetricsTool,
    RegisterModelTool,
    ListModelsTool,
    TrainGradientBoostingTool,
)
from thesiscrew.tools.dataset_tool import (
    BuildKorneuburgDatasetTool,
    BuildFeatureMatrixTool,
)
from thesiscrew.tools.read_input_tool import ReadInputTool
from thesiscrew.tools.report_writer_tool import (
    WriteReportTool,
    ReadReportTool,
    MarkdownTableTool,
    ReportTOCTool,
    RenderMetricsTool,
    ReadArtifactTool,
)

__all__ = [
    # Pegelonline
    "ListStationsTool",
    "StationDetailTool",
    "GetMeasurementsTool",
    "GetMeasurementsCSVTool",
    "GetForecastTool",
    "GetWaterBodiesTool",
    # Open-Meteo
    "HistoricalWeatherTool",
    "ForecastWeatherTool",
    "KorneuburgWeatherTool",
    "KorneuburgForecastTool",
    # eHYD
    "ListAustrianStationsTool",
    "StationMetadataTool",
    "StationDataTool",
    "CharacteristicValuesTool",
    # Data processing
    "ListDataFilesTool",
    "CSVSummaryTool",
    "ParquetSummaryTool",
    "ResampleTool",
    "FillGapsTool",
    "LagFeaturesTool",
    "RollingFeaturesTool",
    "CalendarFeaturesTool",
    "RateOfChangeTool",
    "ChronoSplitTool",
    "ComputeMetricsTool",
    # Model evaluation
    "PersistenceBaselineTool",
    "WalkForwardTool",
    "StratifiedMetricsTool",
    "RegisterModelTool",
    "ListModelsTool",
    "TrainGradientBoostingTool",
    # Dataset
    "BuildKorneuburgDatasetTool",
    "BuildFeatureMatrixTool",
    # Input
    "ReadInputTool",
    # Report
    "WriteReportTool",
    "ReadReportTool",
    "MarkdownTableTool",
    "ReportTOCTool",
    "RenderMetricsTool",
    "ReadArtifactTool",
]