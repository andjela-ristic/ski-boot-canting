import 'dart:convert';
import 'dart:typed_data';

class AnalyzeResult {
  AnalyzeResult({
    required this.sourceName,
    required this.processingTimeMs,
    required this.overlayBytes,
    this.sourcePath,
    this.artifactsDir,
    this.overlayOutputPath,
    this.metadataOutputPath,
  });

  final String sourceName;
  final double processingTimeMs;
  final Uint8List overlayBytes;
  final String? sourcePath;
  final String? artifactsDir;
  final String? overlayOutputPath;
  final String? metadataOutputPath;

  factory AnalyzeResult.fromJson(Map<String, dynamic> json) {
    final overlayDataUrl = json['overlay_data_url'];
    if (overlayDataUrl is! String || overlayDataUrl.isEmpty) {
      throw const FormatException('Missing overlay_data_url in API response.');
    }

    return AnalyzeResult(
      sourceName: (json['video_name'] as String?) ??
          (json['image_name'] as String?) ??
          (json['source_name'] as String?) ??
          'capture.mp4',
      processingTimeMs: (json['processing_time_ms'] as num?)?.toDouble() ?? 0,
      overlayBytes: _decodeDataUrl(overlayDataUrl),
      sourcePath: (json['input_video_path'] as String?) ??
          (json['input_image_path'] as String?) ??
          (json['source_path'] as String?),
      artifactsDir: json['artifacts_dir'] as String?,
      overlayOutputPath: json['overlay_output_path'] as String?,
      metadataOutputPath: json['metadata_output_path'] as String?,
    );
  }

  static Uint8List _decodeDataUrl(String value) {
    final separatorIndex = value.indexOf(',');
    if (separatorIndex < 0 || separatorIndex == value.length - 1) {
      throw const FormatException('overlay_data_url is not a valid data URL.');
    }

    final base64Part = value.substring(separatorIndex + 1);
    return base64Decode(base64Part);
  }
}
