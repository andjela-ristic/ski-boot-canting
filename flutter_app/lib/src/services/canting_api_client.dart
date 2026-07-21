import '../models/analyze_result.dart';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

class CantingApiException implements Exception {
  CantingApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class CantingApiClient {
  const CantingApiClient({
    http.Client? httpClient,
  }) : _httpClient = httpClient;

  final http.Client? _httpClient;

  Future<AnalyzeResult> uploadVideoClip({
    required String baseUrl,
    required String videoPath,
    Duration clipDuration = const Duration(seconds: 2),
    int frameCount = 6,
    bool keepArtifacts = false,
  }) async {
    final normalizedBaseUrl = _normalizeBaseUrl(baseUrl);
    final uri = Uri.parse('$normalizedBaseUrl/frames');
    final client = _httpClient ?? http.Client();
    final ownsClient = _httpClient == null;

    try {
      final file = File(videoPath);
      if (!file.existsSync()) {
        throw CantingApiException('Recorded video file does not exist: $videoPath');
      }

      final request = http.MultipartRequest('POST', uri)
        ..fields['response_mode'] = 'json'
        ..fields['keep_artifacts'] = keepArtifacts.toString()
        ..fields['clip_duration_ms'] = clipDuration.inMilliseconds.toString()
        ..fields['frame_count'] = frameCount.toString();

      request.files.add(
        await http.MultipartFile.fromPath(
          'video',
          videoPath,
          filename: _extractFilename(videoPath),
        ),
      );

      final streamedResponse = await client.send(request).timeout(
            const Duration(seconds: 120),
          );
      final response = await http.Response.fromStream(streamedResponse);
      final payload = _tryDecodeJson(response.body);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        final fallbackMessage = payload?['error'] as String? ??
            (response.body.isNotEmpty
                ? 'API request failed with status ${response.statusCode}: ${response.body}'
                : 'API request failed with status ${response.statusCode}.');
        throw CantingApiException(
          fallbackMessage,
        );
      }

      if (payload == null) {
        throw const FormatException('API response is not valid JSON.');
      }

      return AnalyzeResult.fromJson(payload);
    } on CantingApiException {
      rethrow;
    } on FormatException catch (error) {
      throw CantingApiException('Invalid API response: ${error.message}');
    } catch (error) {
      throw CantingApiException('Could not reach API: $error');
    } finally {
      if (ownsClient) {
        client.close();
      }
    }
  }

  Map<String, dynamic>? _tryDecodeJson(String body) {
    if (body.trim().isEmpty) {
      return null;
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('API response is not a JSON object.');
    }
    return decoded;
  }

  String _normalizeBaseUrl(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      throw CantingApiException('Base URL is required.');
    }

    if (trimmed.endsWith('/')) {
      return trimmed.substring(0, trimmed.length - 1);
    }

    return trimmed;
  }

  String _extractFilename(String path) {
    final normalized = path.replaceAll('\\', '/');
    final segments = normalized.split('/');
    return segments.isEmpty ? 'capture.mp4' : segments.last;
  }
}
