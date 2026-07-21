import 'package:flutter_test/flutter_test.dart';
import 'package:ski_boot_canting_app/src/app.dart';

void main() {
  testWidgets('renders analyzer screen', (WidgetTester tester) async {
    await tester.pumpWidget(const CantingApp());

    expect(find.text('Canting Analyzer'), findsOneWidget);
  });
}
