import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import '../../models/analyze_result.dart';
import '../../services/canting_api_client.dart';
import 'package:flutter/material.dart';

class AnalyzePage extends StatefulWidget {
  const AnalyzePage({super.key});

  @override
  State<AnalyzePage> createState() => _AnalyzePageState();
}

class _AnalyzePageState extends State<AnalyzePage> {
  final _baseUrlController = TextEditingController();
  final _apiClient = const CantingApiClient();

  CameraController? _cameraController;
  List<CameraDescription> _cameras = const [];
  int _selectedCameraIndex = 0;
  bool _isCameraInitializing = true;
  bool _isSubmitting = false;
  bool _isRecording = false;
  String? _errorMessage;
  String _statusMessage = 'Initializing camera...';
  AnalyzeResult? _result;

  @override
  void initState() {
    super.initState();
    _baseUrlController.text = _defaultBaseUrlForPlatform();
    unawaited(_initializeCamera());
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _baseUrlController.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    setState(() {
      _isCameraInitializing = true;
      _errorMessage = null;
      _statusMessage = 'Initializing camera...';
    });

    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw StateError('No camera available on this device.');
      }

      var preferredIndex = 0;
      final backIndex = cameras.indexWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
      );
      if (backIndex >= 0) {
        preferredIndex = backIndex;
      }

      _cameras = cameras;
      await _setActiveCamera(preferredIndex);
    } catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _errorMessage = 'Camera init failed: $error';
        _statusMessage = 'Camera unavailable';
        _isCameraInitializing = false;
      });
    }
  }

  Future<void> _setActiveCamera(int cameraIndex) async {
    final nextController = CameraController(
      _cameras[cameraIndex],
      ResolutionPreset.high,
      enableAudio: false,
    );

    await nextController.initialize();

    try {
      await nextController.prepareForVideoRecording();
    } catch (_) {
      // Some platforms do not require or support an explicit prepare step.
    }

    final previousController = _cameraController;

    if (!mounted) {
      await nextController.dispose();
      return;
    }

    try {
      await previousController?.dispose();
    } catch (_) {
      // Ignore disposal errors on stale controllers.
    }

    setState(() {
      _cameraController = nextController;
      _selectedCameraIndex = cameraIndex;
      _isCameraInitializing = false;
      _statusMessage = 'Ready. Tap "Slikaj" to record 2 seconds.';
    });
  }

  Future<void> _toggleCamera() async {
    if (_cameras.length < 2 || _isSubmitting || _isRecording) {
      return;
    }

    final nextIndex = (_selectedCameraIndex + 1) % _cameras.length;

    setState(() {
      _isCameraInitializing = true;
      _statusMessage = 'Switching camera...';
      _errorMessage = null;
    });

    try {
      await _setActiveCamera(nextIndex);
    } catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _errorMessage = 'Could not switch camera: $error';
        _statusMessage = 'Camera switch failed';
        _isCameraInitializing = false;
      });
    }
  }

  Future<void> _recordAndUpload() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      setState(() {
        _errorMessage = 'Camera is not ready yet.';
      });
      return;
    }

    if (_baseUrlController.text.trim().isEmpty) {
      setState(() {
        _errorMessage = 'Base URL je obavezan.';
      });
      return;
    }

    setState(() {
      _isSubmitting = true;
      _isRecording = true;
      _errorMessage = null;
      _result = null;
      _statusMessage = 'Recording 2-second clip...';
    });

    try {
      await controller.startVideoRecording();
      await Future<void>.delayed(const Duration(seconds: 2));
      final recordedFile = await controller.stopVideoRecording();

      if (!mounted) {
        return;
      }

      setState(() {
        _isRecording = false;
        _statusMessage = 'Uploading clip to POST /frames...';
      });

      final result = await _apiClient.uploadVideoClip(
        baseUrl: _baseUrlController.text,
        videoPath: recordedFile.path,
        clipDuration: const Duration(seconds: 2),
        frameCount: 6,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _result = result;
        _statusMessage = 'Overlay received.';
      });
    } catch (error) {
      if (mounted) {
        setState(() {
          _isRecording = false;
          _statusMessage = 'Recording/upload failed.';
          _errorMessage = error.toString();
        });
      }

      await _safeStopRecording();
    } finally {
      if (mounted) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  Future<void> _safeStopRecording() async {
    final controller = _cameraController;
    if (controller == null) {
      return;
    }

    if (!controller.value.isRecordingVideo) {
      return;
    }

    try {
      await controller.stopVideoRecording();
    } catch (_) {
      // Ignore stop errors after a failed capture flow.
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Canting Analyzer'),
      ),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final horizontalPadding = constraints.maxWidth > 900 ? 32.0 : 16.0;

            return Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 980),
                child: SingleChildScrollView(
                  padding: EdgeInsets.all(horizontalPadding),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Card(
                        clipBehavior: Clip.antiAlias,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            AspectRatio(
                              aspectRatio: _cameraAspectRatio,
                              child: ColoredBox(
                                color: Colors.black,
                                child: _buildCameraSurface(context),
                              ),
                            ),
                            Padding(
                              padding: const EdgeInsets.all(16),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.stretch,
                                children: [
                                  Text(
                                    'One-page camera flow',
                                    style: Theme.of(context).textTheme.headlineSmall,
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    'Klik na "Slikaj" snima 2s video i salje ceo clip na POST /frames.',
                                    style: Theme.of(context).textTheme.bodyMedium,
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    _platformHintText(),
                                    style: Theme.of(context).textTheme.bodySmall,
                                  ),
                                  const SizedBox(height: 16),
                                  TextField(
                                    controller: _baseUrlController,
                                    keyboardType: TextInputType.url,
                                    decoration: InputDecoration(
                                      labelText: 'Base URL',
                                      border: OutlineInputBorder(),
                                      hintText: _defaultBaseUrlForPlatform(),
                                      helperText: _baseUrlHelperText(),
                                    ),
                                  ),
                                  const SizedBox(height: 12),
                                  Container(
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: colorScheme.surfaceContainerHighest,
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    child: Text(_statusMessage),
                                  ),
                                  if (_errorMessage != null) ...[
                                    const SizedBox(height: 12),
                                    Container(
                                      padding: const EdgeInsets.all(12),
                                      decoration: BoxDecoration(
                                        color: colorScheme.errorContainer,
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Text(
                                        _errorMessage!,
                                        style: TextStyle(color: colorScheme.onErrorContainer),
                                      ),
                                    ),
                                  ],
                                  const SizedBox(height: 16),
                                  Row(
                                    children: [
                                      Expanded(
                                        child: FilledButton.icon(
                                          onPressed: (_isSubmitting || _isCameraInitializing)
                                              ? null
                                              : _recordAndUpload,
                                          icon: _isSubmitting
                                              ? const SizedBox(
                                                  width: 18,
                                                  height: 18,
                                                  child: CircularProgressIndicator(strokeWidth: 2),
                                                )
                                              : const Icon(Icons.videocam_rounded),
                                          label: Text(
                                            _isRecording
                                                ? 'Recording...'
                                                : _isSubmitting
                                                    ? 'Uploading...'
                                                    : 'Slikaj',
                                          ),
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      IconButton.filledTonal(
                                        onPressed: (_isSubmitting || _isCameraInitializing || _cameras.length < 2)
                                            ? null
                                            : _toggleCamera,
                                        icon: const Icon(Icons.cameraswitch_rounded),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                      if (_result != null) _ResultCard(result: _result!),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  double get _cameraAspectRatio {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return 16 / 9;
    }
    return controller.value.aspectRatio;
  }

  String _defaultBaseUrlForPlatform() {
    if (kIsWeb) {
      return 'http://127.0.0.1:8000';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'http://10.0.2.2:8000';
      case TargetPlatform.iOS:
        return 'http://127.0.0.1:8000';
      default:
        return 'http://127.0.0.1:8000';
    }
  }

  String _baseUrlHelperText() {
    if (kIsWeb) {
      return 'Docker backend na istoj masini: http://127.0.0.1:8000';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'Android emulator + Docker backend: http://10.0.2.2:8000';
      case TargetPlatform.iOS:
        return 'iOS simulator + Docker backend: http://127.0.0.1:8000';
      default:
        return 'Desktop + Docker backend: http://127.0.0.1:8000';
    }
  }

  String _platformHintText() {
    if (kIsWeb) {
      return 'Web preview koristi CORS-open Docker backend na portu 8000.';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'Android emulator koristi host alias 10.0.2.2 da pogodi Docker backend.';
      case TargetPlatform.iOS:
        return 'iOS simulator koristi localhost/127.0.0.1 za Docker backend na Mac hostu.';
      default:
        return 'Za fizicki uredjaj koristi LAN IP host masine umesto localhost.';
    }
  }

  Widget _buildCameraSurface(BuildContext context) {
    final controller = _cameraController;

    if (_isCameraInitializing) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (controller == null || !controller.value.isInitialized) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'Camera preview unavailable.',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller),
        if (_isRecording)
          const Positioned(
            top: 16,
            right: 16,
            child: _RecordingBadge(),
          ),
      ],
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});

  final AnalyzeResult result;

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Step 09 Overlay',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                _InfoChip(label: 'Source', value: result.sourceName),
                _InfoChip(
                  label: 'Processing',
                  value: '${result.processingTimeMs.toStringAsFixed(2)} ms',
                ),
                if (result.artifactsDir != null)
                  _InfoChip(label: 'Artifacts', value: result.artifactsDir!),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              constraints: const BoxConstraints(minHeight: 240),
              color: Colors.black12,
              alignment: Alignment.center,
              child: Image.memory(
                result.overlayBytes,
                fit: BoxFit.contain,
              ),
            ),
            if (result.sourcePath != null) ...[
              const SizedBox(height: 16),
              SelectableText('Source path: ${result.sourcePath}'),
            ],
            if (result.overlayOutputPath != null) ...[
              const SizedBox(height: 8),
              SelectableText('Overlay output: ${result.overlayOutputPath}'),
            ],
            if (result.metadataOutputPath != null) ...[
              const SizedBox(height: 8),
              SelectableText('Metadata output: ${result.metadataOutputPath}'),
            ],
          ],
        ),
      ),
    );
  }
}

class _RecordingBadge extends StatelessWidget {
  const _RecordingBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.red.shade700,
        borderRadius: BorderRadius.circular(999),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.fiber_manual_record, size: 16, color: Colors.white),
          SizedBox(width: 6),
          Text(
            'REC 2s',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoChip extends StatelessWidget {
  const _InfoChip({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text('$label: $value'),
    );
  }
}
