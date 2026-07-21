import 'features/analyze/analyze_page.dart';
import 'package:flutter/material.dart';

class CantingApp extends StatelessWidget {
  const CantingApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF0F766E);

    return MaterialApp(
      title: 'Ski Boot Canting',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seed),
        useMaterial3: true,
      ),
      home: const AnalyzePage(),
    );
  }
}
