% Alum: Fco Javier Vargas
close all;

if exist('test') && exist('train')
    
    for k = 1:20;
        figure();
        imshow( reshape( test.X(wrong(k), 1:end-1 ), 28, 28) );
        title(['GuessNum: ' num2str(labels(wrong(k))-1) ' RealNum: '  num2str(test.y(wrong(k))-1)]);
        fprintf('k = %d Real Label : %d\n', k, test.y(wrong(k)) - 1);
        fprintf('k = %d Wrong Label: %d\n\n', k, labels(wrong(k)) - 1);
    end

else
    error('Run EjercicioSoftMax before running this script');
end
