#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv;

caminho_para_arquivo = './base.csv';

valores = []

with open(caminho_para_arquivo, 'r') as arquivo_cru:
    arquivo_lido = csv.reader(arquivo_cru)
    
    for linha in arquivo_lido:
        print("---------------------");
        print(linha);
        valores.append(linha);
        