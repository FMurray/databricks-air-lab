package lab;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;

/**
 * Minimal DJL training smoke for AIR: MLP on MNIST.
 * The point is not the model — it's proving a JVM sees the GPU and trains on it
 * through the GA surface (no Docker). GPU placement is automatic when DJL's
 * CUDA-enabled libtorch loads; the PROBE lines make success grep-able in run logs.
 */
public final class TrainMnist {

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 2;

        Engine engine = Engine.getInstance();
        System.out.println("PROBE:jvm=" + System.getProperty("java.version"));
        System.out.println("PROBE:djl_engine=" + engine.getEngineName() + " " + engine.getVersion());
        System.out.println("PROBE:gpu_count=" + engine.getGpuCount());
        System.out.println("PROBE:default_device=" + engine.defaultDevice());

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));

            Mnist mnist = Mnist.builder().setSampling(64, true).build();
            mnist.prepare();

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                            .addEvaluator(new Accuracy())
                            .addTrainingListeners(TrainingListener.Defaults.logging());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(1, 28 * 28));
                EasyTrain.fit(trainer, epochs, mnist, null);
            }
        }
        System.out.println("PROBE:training=done");
    }

    private TrainMnist() {}
}
