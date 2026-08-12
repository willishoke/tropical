import Foundation
import XCTest
@testable import Reversible

final class PatchHierarchyTests: XCTestCase {
    private let engineLibraryFixture = """
    {"schema":"tropical_module_library","schema_version":1,"definitions":[
      {
        "id":"tropical.modal.allpass1","version":1,
        "input":"input","output":"output","input_domain":"modal","output_domain":"modal",
        "parameters":[{"name":"ratio","default":1}],
        "nodes":[
          {"id":"input","kind":"module_input"},
          {"id":"tail","kind":"modal_allpass_tail","params":{"ratio":1},
           "bindings":{"ratio":"ratio"},"in":{"in":["input"]}},
          {"id":"section","kind":"modalmix","in":{"in":["input","tail"]}},
          {"id":"output","kind":"module_output","in":{"in":["section"]}}
        ]
      },
      {
        "id":"tropical.modal.phaser","version":1,
        "input":"input","output":"output","input_domain":"modal","output_domain":"modal",
        "parameters":[
          {"name":"center","default":700},{"name":"sweep","default":1.5},
          {"name":"rate","default":0.2},{"name":"mix","default":0.5}
        ],
        "nodes":[
          {"id":"input","kind":"module_input"},
          {"id":"stage_0","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":0.42044820762685725},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["input"]}},
          {"id":"stage_1","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":0.5946035575013605},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["stage_0"]}},
          {"id":"stage_2","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":0.8408964152537145},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["stage_1"]}},
          {"id":"stage_3","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":1.189207115002721},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["stage_2"]}},
          {"id":"stage_4","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":1.681792830507429},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["stage_3"]}},
          {"id":"stage_5","kind":"module","definition":"tropical.modal.allpass1","definition_version":1,"params":{"ratio":2.378414230005442},"bindings":{"center":"center","sweep":"sweep","rate":"rate"},"in":{"in":["stage_4"]}},
          {"id":"blend","kind":"modalblend","bindings":{"mix":"mix"},"in":{"dry":["input"],"wet":["stage_5"]}},
          {"id":"output","kind":"module_output","in":{"in":["blend"]}}
        ]
      }
    ]}
    """

    private func engineLibrary() throws -> [ModuleReference: ModuleDefinitionState] {
        let raw = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(engineLibraryFixture.utf8)
        )
        return try EngineModuleLibrary.decode(raw)
    }

    func testEngineOwnedModuleLibraryDecodesWithoutASecondSemanticTable() throws {
        let library = try engineLibrary()
        let allpass = try XCTUnwrap(library[StandardModuleLibrary.allpass])

        XCTAssertEqual(allpass.parameters, [.init(name: "ratio", defaultValue: 1)])
        XCTAssertEqual(allpass.graph.order, ["input", "tail", "section", "output"])
        XCTAssertEqual(allpass.graph.nodes["tail"]?.kind, .modalTail)
        XCTAssertEqual(allpass.graph.nodes["tail"]?.bindings, ["ratio": "ratio"])
    }

    func testStandardPhaserDecomposesIntoSixEditableAllpassInstances() throws {
        let definition = try XCTUnwrap(try engineLibrary()[StandardModuleLibrary.phaser])
        let stages = definition.graph.order.compactMap { definition.graph.nodes[$0] }
            .filter { $0.kind == .allpass }

        XCTAssertEqual(stages.count, 6)
        XCTAssertEqual(
            stages.compactMap(\.module),
            Array(repeating: StandardModuleLibrary.allpass, count: 6)
        )
        XCTAssertEqual(
            stages.compactMap { $0.bindings["center"] },
            Array(repeating: "center", count: 6)
        )
        XCTAssertEqual(stages.compactMap { $0.values["ratio"] }, [
            0.42044820762685725, 0.5946035575013605, 0.8408964152537145,
            1.189207115002721, 1.681792830507429, 2.378414230005442,
        ])
        XCTAssertEqual(
            definition.graph.nodes["blend"]?.inputs,
            ["dry": ["input"], "wet": ["stage_5"]]
        )
    }

    func testStandardAllpassExposesDirectPlusTailTopology() throws {
        let definition = try XCTUnwrap(try engineLibrary()[StandardModuleLibrary.allpass])
        XCTAssertEqual(definition.graph.nodes["tail"]?.kind, .modalTail)
        XCTAssertEqual(definition.graph.nodes["tail"]?.inputs["in"], ["input"])
        XCTAssertEqual(definition.graph.nodes["section"]?.kind, .modalmix)
        XCTAssertEqual(
            definition.graph.nodes["section"]?.inputs["in"],
            ["input", "tail"]
        )
        XCTAssertEqual(definition.graph.nodes["output"]?.inputs["in"], ["section"])
    }

    func testV1PhaserMigratesDeterministicallyToInstalledV3Definition() throws {
        let source = """
        {
          "version": 1,
          "nodes": [
            {"id":"res1","kind":"resonator","x":20,"y":30,"hue":30,
             "values":{"freq":220,"decay":4},"inputs":{"addr":[]}},
            {"id":"ph2","kind":"phaser","x":120,"y":30,"hue":167,
             "values":{"center":810,"sweep":1.75,"rate":0.31,"mix":0.42},
             "inputs":{"in":["res1"]}}
          ],
          "order":["res1","ph2"],"velocity":-1,
          "panX":11,"panY":-7,"autoArrange":true
        }
        """
        let first = try JSONDecoder().decode(PatchDocument.self, from: Data(source.utf8))
        let second = try JSONDecoder().decode(PatchDocument.self, from: Data(source.utf8))

        XCTAssertEqual(first.version, 3)
        XCTAssertEqual(first.nodes.first { $0.id == "ph2" }?.module, StandardModuleLibrary.phaser)
        XCTAssertTrue(first.definitions.isEmpty)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        XCTAssertEqual(
            try encoder.encode(first),
            try encoder.encode(second)
        )
    }

    func testV3RoundTripPreservesDefinitionOrderBindingsAndPresentation() throws {
        let installed = try XCTUnwrap(try engineLibrary()[StandardModuleLibrary.phaser])
        let phaser = StandardModuleLibrary.detached(installed, instancePath: ["ph2"])
        let definition = PatchDocument.Definition(
            id: phaser.reference.id,
            version: phaser.reference.version,
            title: phaser.title,
            input: phaser.inputNodeID,
            output: phaser.outputNodeID,
            parameters: phaser.parameters,
            graph: .init(
                nodes: phaser.graph.order.compactMap { phaser.graph.nodes[$0] }.map { node in
                    .init(
                        id: node.id, kind: node.kind,
                        x: node.position.x, y: node.position.y, hue: node.hue,
                        values: node.values, inputs: node.inputs,
                        module: node.module, bindings: node.bindings
                    )
                },
                order: phaser.graph.order,
                panX: 91,
                panY: -24
            )
        )
        let document = PatchDocument(
            nodes: [
                .init(
                    id: "ph2", kind: .phaser, x: 120, y: 30, hue: 167,
                    values: ["center": 810, "sweep": 1.75, "rate": 0.31, "mix": 0.42],
                    inputs: ["in": []], module: phaser.reference
                ),
            ],
            order: ["ph2"], velocity: -1,
            panX: 11, panY: -7, autoArrange: true,
            definitions: [definition]
        )

        let decoded = try JSONDecoder().decode(
            PatchDocument.self,
            from: JSONEncoder().encode(document)
        )
        XCTAssertEqual(decoded.version, 3)
        XCTAssertEqual(decoded.definitions.first?.graph.order, phaser.graph.order)
        XCTAssertEqual(
            decoded.definitions.first?.graph.nodes.first { $0.id == "stage_0" }?.bindings,
            ["center": "center", "sweep": "sweep", "rate": "rate"]
        )
        XCTAssertEqual(decoded.definitions.first?.graph.panX, 91)
        XCTAssertEqual(decoded.nodes.first?.module, phaser.reference)
    }
}
